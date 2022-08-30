use thiserror::Error;

#[derive(Error, Debug)]
pub enum BDFError {
    #[error("Couldn't start Vulkano: {0}")]
    InstanceCreationError(#[from] vulkano::instance::InstanceCreationError),
    #[error("No Vulkan physical devices found.")]
    NoPhysicalDevices,
    #[error("No Vulkan queue families found.")]
    NoQueueFamilies,
    #[error("No Vulkan queue families supporting compute shaders found")]
    NoComputeShaders,
    #[error("Couldn't use Vulkan device {0}")]
    DeviceCreationError(#[from] vulkano::device::DeviceCreationError),
    #[error("Couldn't create shader: {0}")]
    ShaderCreationError(#[from] vulkano::shader::ShaderCreationError),
    #[error("Couldn't create compute pipeline: {0}")]
    ComputePipelineCreationError(#[from] vulkano::pipeline::compute::ComputePipelineCreationError),
    #[error("Couldn't dispatch compute task on device: {0}")]
    DispatchError(#[from] vulkano::command_buffer::DispatchError),
    #[error("Couldn't execute command buffer: {0}")]
    CommandBufferExecError(#[from] vulkano::command_buffer::CommandBufferExecError),
    #[error("Couldn't build command buffer: {0}")]
    CommandBufferBuildError(#[from] vulkano::command_buffer::BuildError),
    #[error("Couldn't flush command buffer: {0}")]
    CommandBufferFlushError(#[from] vulkano::sync::FlushError),
    #[error("Couldn't allocate device memory: {0}")]
    DeviceMemoryAllocationError(#[from] vulkano::memory::DeviceMemoryAllocationError),
    #[error("Error acquiring read lock on device buffer: {0}")]
    BufferReadLockError(#[from] vulkano::buffer::cpu_access::ReadLockError),
}

pub type Result<X> = std::result::Result<X, BDFError>;
