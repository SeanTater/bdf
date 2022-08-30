use std::sync::Arc;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::shader::ShaderModule;
use vulkano::sync::{self, GpuFuture};
use crate::errors::{BDFError, Result};

/// The Vulkano context, representing one device (probably a GPU) and one queue (for compute).
pub struct Accelerator {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
}
impl Accelerator {
    pub fn new() -> Result<Self> {
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

    pub fn device(&self) -> Arc<Device> {
        self.device.clone()
    }

    pub fn queue(&self) -> Arc<Queue> {
        self.queue.clone()
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
    /// Creates a new code object from the given shader module.
    /// Don't use this directly, instead use Expression
    pub fn new(accel: &'a Accelerator, shader: Arc<ShaderModule>) -> Result<Self> {
        Ok(Code {
            accel,
            shader
        })
    }
    pub fn run(&self, parameters: &[&Tensor<u32>]) -> Result<()> {
        // Prep the pipeline to run the code
        let compute_pipeline = ComputePipeline::new(
            self.accel.device(),
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
            self.accel.device(),
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
        let future = sync::now(self.accel.device())
            .then_execute(self.accel.queue(), command_buffer)?
            .then_signal_fence_and_flush()?;

        // Await it
        future.wait(None)?;
        Ok(())
    }
}