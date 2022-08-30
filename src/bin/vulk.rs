use std::collections::HashMap;
use std::sync::Arc;

use bdf::errors::{BDFError, Result};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::shader::ShaderModule;
use vulkano::sync::{self, GpuFuture};

/// The Vulkano context, representing one device (probably a GPU) and one queue (for compute).
pub struct Accelerator {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
}
impl Accelerator {
    fn new() -> Result<Self> {
        let instance = Instance::new(InstanceCreateInfo::default())?;
        let physical = PhysicalDevice::enumerate(&instance)
            .next()
            .ok_or(BDFError::NoPhysicalDevices)?;
        if physical.queue_families().count() == 0 {
            return Err(BDFError::NoQueueFamilies);
        }
        let queue_family = physical
            .queue_families()
            .find(|&q| q.supports_compute())
            .ok_or(BDFError::NoComputeShaders)?;
        let (device, mut queues) = Device::new(
            physical,
            DeviceCreateInfo {
                // here we pass the desired queue families that we want to use
                queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
                ..Default::default()
            },
        )?;
        let queue = queues.next().unwrap();
        Ok(Accelerator {
            instance,
            device,
            queue,
        })
    }

    pub fn code(&self) -> Result<Code> {
        Code::new(&self)
    }
}

/// A closed trait including all primitives a GPU can use.
pub trait Primitive {
    fn zero() -> Self;
}
impl Primitive for f32 {
    fn zero() -> Self {
        0.0
    }
}
impl Primitive for u32 {
    fn zero() -> Self {
        0
    }
}

/// An N-dimensional array, usable for reading or writing, from the GPU.
pub struct Tensor<T>
where
    [T]: vulkano::buffer::BufferContents,
    T: Primitive + Clone,
{
    buffer: Arc<CpuAccessibleBuffer<[T]>>,
}

impl<T> Tensor<T>
where
    [T]: vulkano::buffer::BufferContents,
    T: Primitive + Clone,
{
    /// Creates a new 1 dimensional tensor with the given number of elements.
    pub fn zeros(accel: &Accelerator, size: usize) -> Result<Self> {
        Self::from_iter(accel, (0..size).map(|_| T::zero()))
    }

    /// Creates a new 1 dimensional tensor with the given number of elements.
    pub fn from_iter<X: ExactSizeIterator<Item = T>>(accel: &Accelerator, it: X) -> Result<Self> {
        Ok(Tensor {
            buffer: CpuAccessibleBuffer::from_iter(
                accel.device.clone(),
                BufferUsage::all(),
                false, // Todo: Is this necessary?
                it,
            )?,
        })
    }

    /// Copies the device data into a host-accessible array.
    pub fn read(&self) -> Result<Vec<T>> {
        Ok(self.buffer.read()?.to_vec())
    }
}

pub struct Code<'a> {
    accel: &'a Accelerator,
    shader: Arc<ShaderModule>,
}
impl<'a> Code<'a> {
    pub fn new(accel: &'a Accelerator) -> Result<Self> {
        let shader = cs::load(accel.device.clone())?;
        Ok(Code { accel, shader })
    }

    pub fn run(&self, parameters: &[&Tensor<u32>]) -> Result<()> {
        // Prep the pipeline to run the code
        let compute_pipeline = ComputePipeline::new(
            self.accel.device.clone(),
            self.shader.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )?;

        // Descriptor sets are like parameter bindings for functions
        let layout = compute_pipeline
            .layout()
            .set_layouts()
            .get(0)
            .expect("Compute pipeline didn't have any descriptor sets");
        let set = PersistentDescriptorSet::new(
            layout.clone(),
            //[WriteDescriptorSet::buffer(0, tensor.buffer.clone())], // 0 is the binding
            parameters
                .iter()
                .enumerate()
                .map(|(ix, tensor)| WriteDescriptorSet::buffer(ix as u32, tensor.buffer.clone())),
        )
        .unwrap();

        // A command buffer for the pipeline
        let mut builder = AutoCommandBufferBuilder::primary(
            self.accel.device.clone(),
            self.accel.queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .bind_pipeline_compute(compute_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                compute_pipeline.layout().clone(),
                0, // 0 is the index of our set
                set,
            )
            .dispatch([1024, 1, 1])?; // TODO: Allow choosing this

        let command_buffer = builder.build()?;

        // Run an async on the GPU
        let future = sync::now(self.accel.device.clone())
            .then_execute(self.accel.queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?;

        // Await it
        future.wait(None)?;
        Ok(())
    }
}

fn main() -> Result<()> {
    // Get the device
    let accel = Accelerator::new()?;
    let code = accel.code()?;

    // The data you're going to send it
    //let data_iter = 0..65536;
    //let data_buffer = Tensor::zeros(&accel, 65536)?;
    let data_buffer = Tensor::from_iter(&accel, (0..65536).map(|x| x as u32))?;
    // let data_buffer =
    //     CpuAccessibleBuffer::from_iter(accel.device.clone(), BufferUsage::all(), false, data_iter)
    //         .expect("failed to create buffer");

    code.run(&[&data_buffer, &data_buffer])?;

    let content = data_buffer.read().unwrap();
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, n as u32 * 12 + 5);
    }

    println!("Everything succeeded!");
    Ok(())
}

//
// Format GLSL code to do basic vector arithmetic operations on primitive datatypes.
//

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    //F16,
    F32,
    F64,
    //I8,
    //I16,
    I32,
    //I64,
    //U8,
    //U16,
    U32,
    //U64,
    Bool,
}
impl Dtype {
    pub fn glsl_type(&self) -> &'static str {
        match self {
            //Dtype::F16 => "half",
            Dtype::F32 => "float",
            Dtype::F64 => "double",
            //Dtype::I8 => "int8_t",
            //Dtype::I16 => "int16_t",
            Dtype::I32 => "int",
            //Dtype::I64 => "int64_t",
            //Dtype::U8 => "uint8_t",
            //Dtype::U16 => "uint16_t",
            Dtype::U32 => "uint",
            //Dtype::U64 => "uint64_t",
            Dtype::Bool => "bool",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    And,
    Or,
    Xor,
}
impl Operation {
    pub fn name(&self) -> &'static str {
        match self {
            Operation::Add => "add",
            Operation::Sub => "sub",
            Operation::Mul => "mul",
            Operation::Div => "div",
            Operation::Mod => "mod",
            Operation::And => "and",
            Operation::Or => "or",
            Operation::Xor => "xor",
        }
    }
    pub fn glsl(&self) -> &'static str {
        match self {
            Operation::Add => "+",
            Operation::Sub => "-",
            Operation::Mul => "*",
            Operation::Div => "/",
            Operation::Mod => "%",
            Operation::And => "&",
            Operation::Or => "|",
            Operation::Xor => "^",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expression {
    Constant {
        val: String,
        dtype: Dtype,
    },
    Variable {
        name: String,
        dtype: Dtype,
    },
    Operation {
        op: Operation,
        dtype: Dtype,
        lhs: Box<Expression>,
        rhs: Box<Expression>,
    },
}
impl Expression {
    /// Get the datatype of the expression.
    pub fn dtype(&self) -> Dtype {
        match self {
            Expression::Constant { dtype, .. } => *dtype,
            Expression::Variable { dtype, .. } => *dtype,
            Expression::Operation { dtype, .. } => *dtype,
        }
    }

    /// Assign bindings to each variable in the expression, using a vector
    pub fn variables(&self, bindings: &mut Vec<(String, Dtype)>) {
        match self {
            Expression::Constant { .. } => {}
            Expression::Variable { name, dtype } => {
                // It's an error if the variable is already bound with a different type
                if let Some(existing) = bindings.iter().find(|(n, _)| name == n) {
                    if *dtype != existing.1 {
                        panic!("Variable {} already bound with different type", name);
                    }
                } else {
                    bindings.push((name.clone(), *dtype));
                }
            }
            Expression::Operation { lhs, rhs, .. } => {
                lhs.variables(bindings);
                rhs.variables(bindings);
            }
        }
    }

    /// Format the expression as a GLSL expression. (Not the whole shader)
    pub fn glsl_expression(&self) -> String {
        match self {
            Expression::Constant { val, .. } => format!("{}", val),
            Expression::Variable { name, .. } => format!("{}.data[gl_GlobalInvocationID.x]", name),
            Expression::Operation { op, lhs, rhs, .. } => {
                format!(
                    "({} {} {})",
                    lhs.glsl_expression(),
                    op.glsl(),
                    rhs.glsl_expression()
                )
            }
        }
    }

    /// Format GLSL bindings for the expression.
    pub fn glsl_bindings(&self) -> String {
        let mut bindings = Vec::new();
        self.variables(&mut bindings);
        // Append an output binding for the result of the expression.
        bindings.push(("dest".to_string(), self.dtype()));

        let mut result = String::new();
        for (index, (name, dtype)) in bindings.iter().enumerate() {
            result.push_str(&format!(
                "layout(set = 0, binding = {}) buffer {} {{
    {} data[];
}} {};
",
                index,
                name.to_ascii_uppercase(),
                dtype.glsl_type(),
                name
            ));
        }
        result
    }

    /// Format a new shader with the expression as a GLSL shader source code
    pub fn glsl_shader_source(&self) -> String {
        format!(
            "#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
{layouts}
void main() {{
    dest.data[gl_GlobalInvocationID.x] = {expression};
}}",
            layouts = self.glsl_bindings(),
            expression = self.glsl_expression(),
        )
    }

    /// Compile the expression as a GLSL shader.
    /// Returns the vulkano_shaders::Shader module.
    pub fn compile<'t>(&self, accel: &'t Accelerator) -> Result<Code<'t>> {
        let compiler = shaderc::Compiler::new().ok_or(BDFError::ShaderCCompilerError)?;

        let mut variables = vec![];
        self.variables(&mut variables);
        let param_count = variables.len();

        let source = self.glsl_shader_source();
        let options = shaderc::CompileOptions::new().ok_or(BDFError::ShaderCCompilerError)?;
        let compiled = compiler.compile_into_spirv(
            &source,
            shaderc::ShaderKind::Compute,
            "expression.glsl",
            "main",
            Some(&options),
        )?;
        let spirv_words = compiled.as_binary();
        let shader = bdf::builder::load(accel.device.clone(), spirv_words, param_count+1)?;

        Ok(Code { accel, shader })
    }
}

#[test]
fn test_basic_expression() {
    let expr = Expression::Operation {
        op: Operation::Add,
        dtype: Dtype::F32,
        lhs: Box::new(Expression::Constant {
            val: "1.0".to_string(),
            dtype: Dtype::F32,
        }),
        rhs: Box::new(Expression::Constant {
            val: "2.0".to_string(),
            dtype: Dtype::F32,
        }),
    };
    assert_eq!(expr.glsl_expression(), "(1.0 + 2.0)");
}

#[test]
fn test_nested_expression() {
    let expr = Expression::Operation {
        op: Operation::Add,
        dtype: Dtype::F32,
        lhs: Box::new(Expression::Operation {
            op: Operation::Add,
            dtype: Dtype::F32,
            lhs: Box::new(Expression::Constant {
                val: "1.0".to_string(),
                dtype: Dtype::F32,
            }),
            rhs: Box::new(Expression::Constant {
                val: "2.0".to_string(),
                dtype: Dtype::F32,
            }),
        }),
        rhs: Box::new(Expression::Constant {
            val: "3.0".to_string(),
            dtype: Dtype::F32,
        }),
    };
    assert_eq!(expr.glsl_expression(), "((1.0 + 2.0) + 3.0)");
}

#[test]
fn test_expression_variables() {
    let expr = Expression::Operation {
        op: Operation::Add,
        dtype: Dtype::F32,
        lhs: Box::new(Expression::Variable {
            name: "a".to_string(),
            dtype: Dtype::F32,
        }),
        rhs: Box::new(Expression::Variable {
            name: "b".to_string(),
            dtype: Dtype::F32,
        }),
    };
    let mut bindings = Vec::new();
    expr.variables(&mut bindings);
    assert_eq!(
        bindings,
        vec![("a".to_string(), Dtype::F32), ("b".to_string(), Dtype::F32)]
    );
}

#[test]
fn test_expression_bindings() {
    let expr = Expression::Operation {
        op: Operation::Add,
        dtype: Dtype::F32,
        lhs: Box::new(Expression::Variable {
            name: "a".to_string(),
            dtype: Dtype::F32,
        }),
        rhs: Box::new(Expression::Variable {
            name: "b".to_string(),
            dtype: Dtype::F32,
        }),
    };
    assert_eq!(
        expr.glsl_bindings(),
        "layout(set = 0, binding = 0) buffer A {
    float data[];
} a;
layout(set = 0, binding = 1) buffer B {
    float data[];
} b;
layout(set = 0, binding = 2) buffer DEST {
    float data[];
} dest;
"
    );
}

#[test]
fn test_expression_shader() {
    let expr = Expression::Operation {
        op: Operation::Add,
        dtype: Dtype::F32,
        lhs: Box::new(Expression::Variable {
            name: "a".to_string(),
            dtype: Dtype::F32,
        }),
        rhs: Box::new(Expression::Variable {
            name: "b".to_string(),
            dtype: Dtype::F32,
        }),
    };
    assert_eq!(
        expr.glsl_shader_source(),
        "#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) buffer A {
    float data[];
} a;
layout(set = 0, binding = 1) buffer B {
    float data[];
} b;
layout(set = 0, binding = 2) buffer DEST {
    float data[];
} dest;

void main() {
    dest.data[gl_GlobalInvocationID.x] = (a.data[gl_GlobalInvocationID.x] + b.data[gl_GlobalInvocationID.x]);
}"
    );
}

#[test]
fn test_compile_run_shader() {
    let expr = Expression::Operation {
        op: Operation::Add,
        dtype: Dtype::U32,
        lhs: Box::new(Expression::Variable {
            name: "a".to_string(),
            dtype: Dtype::U32,
        }),
        rhs: Box::new(Expression::Variable {
            name: "b".to_string(),
            dtype: Dtype::U32,
        }),
    };
    let ref accel = Accelerator::new().unwrap();
    let shader = expr.compile(accel).unwrap();
    let buf_a = Tensor::from_iter(accel, 0..10).unwrap();
    let buf_b = Tensor::from_iter(accel, 20..30).unwrap();
    let buf_out = Tensor::zeros(accel, 10).unwrap();
    shader.run(&[&buf_a, &buf_b, &buf_out]).unwrap();
    assert_eq!(
        buf_out.read().unwrap(),
        vec![20, 22, 24, 26, 28, 30, 32, 34, 36, 38]
    );
}

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer SRC {
    uint data[];
} src;

layout(set = 0, binding = 1) buffer DEST {
    uint data[];
} dest;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    dest.data[idx] = src.data[idx] * 12 + 5;
}"
    }
}
