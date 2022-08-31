//!
//! Format GLSL code to do basic vector arithmetic operations on primitive datatypes.
//!
use std::ops::{Add, Sub, Rem, Div, Mul, Neg};

use crate::errors::{BDFError, Result};
use crate::device::{Accelerator, Code, Tensor};

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
pub enum Expr {
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
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
}
impl Expr {
    /// Get the datatype of the expression.
    pub fn dtype(&self) -> Dtype {
        match self {
            Expr::Constant { dtype, .. } => *dtype,
            Expr::Variable { dtype, .. } => *dtype,
            Expr::Operation { dtype, .. } => *dtype,
        }
    }

    /// Create an expression from a variable name
    pub fn var<X: Into<String>>(name: X, dtype: Dtype) -> Expr {
        Expr::Variable { name: name.into(), dtype }
    }

    /// Assign bindings to each variable in the expression, using a vector
    pub fn variables(&self, bindings: &mut Vec<(String, Dtype)>) {
        match self {
            Expr::Constant { .. } => {}
            Expr::Variable { name, dtype } => {
                // It's an error if the variable is already bound with a different type
                if let Some(existing) = bindings.iter().find(|(n, _)| name == n) {
                    if *dtype != existing.1 {
                        panic!("Variable {} already bound with different type", name);
                    }
                } else {
                    bindings.push((name.clone(), *dtype));
                }
            }
            Expr::Operation { lhs, rhs, .. } => {
                lhs.variables(bindings);
                rhs.variables(bindings);
            }
        }
    }

    /// Format the expression as a GLSL expression. (Not the whole shader)
    pub fn glsl_expression(&self) -> String {
        match self {
            Expr::Constant { val, .. } => format!("{}", val),
            Expr::Variable { name, .. } => format!("{}.data[gl_GlobalInvocationID.x]", name),
            Expr::Operation { op, lhs, rhs, .. } => {
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
        let shader = crate::compile::load(accel.device(), spirv_words, param_count + 1)?;

        Ok(Code::new(accel, shader)?)
    }
}

// Implementations for arithmetic operations on references to Exprs.

impl<T: Into<Expr>> Add<T> for &Expr {
    type Output = Expr;
    fn add(self, other: T) -> Expr {
        Expr::Operation {
            op: Operation::Add,
            dtype: self.dtype(),
            lhs: Box::new(self.clone()),
            rhs: Box::new(other.into()),
        }
    }
}
impl<T: Into<Expr>> Sub<T> for &Expr {
    type Output = Expr;
    fn sub(self, other: T) -> Expr {
        Expr::Operation {
            op: Operation::Sub,
            dtype: self.dtype(),
            lhs: Box::new(self.clone()),
            rhs: Box::new(other.into()),
        }
    }
}
impl<T: Into<Expr>> Mul<T> for &Expr {
    type Output = Expr;
    fn mul(self, other: T) -> Expr {
        Expr::Operation {
            op: Operation::Mul,
            dtype: self.dtype(),
            lhs: Box::new(self.clone()),
            rhs: Box::new(other.into()),
        }
    }
}
impl<T: Into<Expr>> Div<T> for &Expr {
    type Output = Expr;
    fn div(self, other: T) -> Expr {
        Expr::Operation {
            op: Operation::Div,
            dtype: self.dtype(),
            lhs: Box::new(self.clone()),
            rhs: Box::new(other.into()),
        }
    }
}
impl<T: Into<Expr>> Rem<T> for &Expr {
    type Output = Expr;
    fn rem(self, other: T) -> Expr {
        Expr::Operation {
            op: Operation::Mod,
            dtype: self.dtype(),
            lhs: Box::new(self.clone()),
            rhs: Box::new(other.into()),
        }
    }
}
impl Neg for &Expr {
    type Output = Expr;
    fn neg(self) -> Expr {
        Expr::Operation {
            op: Operation::Sub,
            dtype: self.dtype(),
            lhs: Box::new(Expr::Constant {
                val: "0".into(),
                dtype: self.dtype(),
            }),
            rhs: Box::new(self.clone()),
        }
    }
}

// Implementations for arithmetic operations on Exprs.

impl<T: Into<Expr>> Add<T> for Expr {
    type Output = Expr;
    fn add(self, other: T) -> Expr {
        &self + other.into()
    }
}
impl<T: Into<Expr>> Sub<T> for Expr {
    type Output = Expr;
    fn sub(self, other: T) -> Expr {
        &self - other.into()
    }
}
impl<T: Into<Expr>> Mul<T> for Expr {
    type Output = Expr;
    fn mul(self, other: Expr) -> Expr {
        &self * other.into()
    }
}
impl<T: Into<Expr>> Div<T> for Expr {
    type Output = Expr;
    fn div(self, other: T) -> Expr {
        &self / other.into()
    }
}
impl<T: Into<Expr>> Rem<T> for Expr {
    type Output = Expr;
    fn rem(self, other: T) -> Expr {
        &self % other.into()
    }
}
impl Neg for Expr {
    type Output = Expr;
    fn neg(self) -> Expr {
        -&self
    }
}

// Convert primitive types to the corresponding expression type.
impl From<i32> for Expr {
    fn from(val: i32) -> Expr {
        Expr::Constant {
            val: format!("{}", val),
            dtype: Dtype::I32,
        }
    }
}
impl From<u32> for Expr {
    fn from(val: u32) -> Expr {
        Expr::Constant {
            val: format!("{}", val),
            dtype: Dtype::U32,
        }
    }
}
impl From<f32> for Expr {
    fn from(val: f32) -> Expr {
        Expr::Constant {
            val: format!("{}", val),
            dtype: Dtype::F32,
        }
    }
}
impl From<f64> for Expr {
    fn from(val: f64) -> Expr {
        Expr::Constant {
            val: format!("{}", val),
            dtype: Dtype::F64,
        }
    }
}
impl From<bool> for Expr {
    fn from(val: bool) -> Expr {
        Expr::Constant {
            val: format!("{}", val),
            dtype: Dtype::Bool,
        }
    }
}
impl From<&Expr> for Expr {
    fn from(val: &Expr) -> Expr {
        val.clone()
    }
}


#[test]
fn test_basic_expression() {
    let expr = Expr::Operation {
        op: Operation::Add,
        dtype: Dtype::F32,
        lhs: Box::new(Expr::Constant {
            val: "1.0".to_string(),
            dtype: Dtype::F32,
        }),
        rhs: Box::new(Expr::Constant {
            val: "2.0".to_string(),
            dtype: Dtype::F32,
        }),
    };
    assert_eq!(expr.glsl_expression(), "(1.0 + 2.0)");
}

#[test]
fn test_nested_expression() {
    let expr = Expr::Operation {
        op: Operation::Add,
        dtype: Dtype::F32,
        lhs: Box::new(Expr::Operation {
            op: Operation::Add,
            dtype: Dtype::F32,
            lhs: Box::new(Expr::Constant {
                val: "1.0".to_string(),
                dtype: Dtype::F32,
            }),
            rhs: Box::new(Expr::Constant {
                val: "2.0".to_string(),
                dtype: Dtype::F32,
            }),
        }),
        rhs: Box::new(Expr::Constant {
            val: "3.0".to_string(),
            dtype: Dtype::F32,
        }),
    };
    assert_eq!(expr.glsl_expression(), "((1.0 + 2.0) + 3.0)");
}

#[test]
fn test_expression_variables() {
    let expr = Expr::Operation {
        op: Operation::Add,
        dtype: Dtype::F32,
        lhs: Box::new(Expr::Variable {
            name: "a".to_string(),
            dtype: Dtype::F32,
        }),
        rhs: Box::new(Expr::Variable {
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
    let expr = Expr::Operation {
        op: Operation::Add,
        dtype: Dtype::F32,
        lhs: Box::new(Expr::Variable {
            name: "a".to_string(),
            dtype: Dtype::F32,
        }),
        rhs: Box::new(Expr::Variable {
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
    let expr = Expr::Operation {
        op: Operation::Add,
        dtype: Dtype::F32,
        lhs: Box::new(Expr::Variable {
            name: "a".to_string(),
            dtype: Dtype::F32,
        }),
        rhs: Box::new(Expr::Variable {
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
    let expr = Expr::Operation {
        op: Operation::Add,
        dtype: Dtype::U32,
        lhs: Box::new(Expr::Variable {
            name: "a".to_string(),
            dtype: Dtype::U32,
        }),
        rhs: Box::new(Expr::Variable {
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

#[test]
fn test_expression_arithmetic() {
    let lhs = Expr::Variable {
        name: "a".to_string(),
        dtype: Dtype::U32,
    };
    let rhs = Expr::Variable {
        name: "b".to_string(),
        dtype: Dtype::U32,
    };
    // Addition
    assert_eq!(&lhs + &rhs, Expr::Operation {
        op: Operation::Add,
        dtype: Dtype::U32,
        lhs: Box::new(lhs.clone()),
        rhs: Box::new(rhs.clone()),
    });
    // Subtraction
    assert_eq!(&lhs - &rhs, Expr::Operation {
        op: Operation::Sub,
        dtype: Dtype::U32,
        lhs: Box::new(lhs.clone()),
        rhs: Box::new(rhs.clone()),
    });
    // Multiplication
    assert_eq!(&lhs * &rhs, Expr::Operation {
        op: Operation::Mul,
        dtype: Dtype::U32,
        lhs: Box::new(lhs.clone()),
        rhs: Box::new(rhs.clone()),
    });
    // Division
    assert_eq!(&lhs / &rhs, Expr::Operation {
        op: Operation::Div,
        dtype: Dtype::U32,
        lhs: Box::new(lhs.clone()),
        rhs: Box::new(rhs.clone()),
    });
    // Modulus
    assert_eq!(&lhs % &rhs, Expr::Operation {
        op: Operation::Mod,
        dtype: Dtype::U32,
        lhs: Box::new(lhs.clone()),
        rhs: Box::new(rhs.clone()),
    });
    // Negation
    assert_eq!(-&lhs, Expr::Operation {
        op: Operation::Sub,
        dtype: Dtype::U32,
        lhs: Box::new(Expr::Constant {
            val: "0".into(),
            dtype: Dtype::U32,
        }),
        rhs: Box::new(lhs.clone()),
    });
}

#[test]
fn test_expression_literals() {
    assert_eq!(Expr::from(1u32), Expr::Constant {
        val: "1".into(),
        dtype: Dtype::U32,
    });
    assert_eq!(Expr::from(1i32), Expr::Constant {
        val: "1".into(),
        dtype: Dtype::I32,
    });
    assert_eq!(Expr::from(1f32), Expr::Constant {
        val: "1".into(),
        dtype: Dtype::F32,
    });
    assert_eq!(Expr::from(1f64), Expr::Constant {
        val: "1".into(),
        dtype: Dtype::F64,
    });
    assert_eq!(Expr::from(true), Expr::Constant {
        val: "true".into(),
        dtype: Dtype::Bool,
    });
}

#[test]
fn test_complex_high_level_expression() {
    let expr = Expr::var("a", Dtype::U32) + Expr::var("b", Dtype::U32) * 12u32 + 5u32;
    assert_eq!( expr, 
        Expr::Operation {
            op: Operation::Add,
            dtype: Dtype::U32,
            lhs: Box::new(Expr::Variable { name: "a".into(), dtype: Dtype::U32 }),
            rhs: Box::new(Expr::Operation {
                op: Operation::Add,
                dtype: Dtype::U32,
                lhs: Box::new(Expr::Operation {
                    op: Operation::Mul,
                    dtype: Dtype::U32,
                    lhs: Box::new(Expr::Constant {
                        val: "12".into(),
                        dtype: Dtype::U32,
                    }),
                    rhs: Box::new(Expr::Variable { name: "b".into(), dtype: Dtype::U32 }),
                }),
                rhs: Box::new(Expr::Constant {
                    val: "5".into(),
                    dtype: Dtype::U32,
                }),
            }),
        }
    );
}