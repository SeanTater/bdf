/// Loads the shader in Vulkan as a `ShaderModule`.
#[inline]
#[allow(unsafe_code)]
pub fn load(
    device: ::std::sync::Arc<::vulkano::device::Device>,
    spirv_words: &[u32],
    param_count: usize,
) -> Result<::std::sync::Arc<::vulkano::shader::ShaderModule>, ::vulkano::shader::ShaderCreationError>
{
    let _bytes = ();
    let req = ::vulkano::shader::DescriptorRequirements {
        descriptor_types: vec![
            ::vulkano::descriptor_set::layout::DescriptorType::StorageBuffer,
            ::vulkano::descriptor_set::layout::DescriptorType::StorageBufferDynamic,
        ],
        descriptor_count: 1u32,
        image_format: None,
        image_multisampled: false,
        image_scalar_type: None,
        image_view_type: None,
        sampler_compare: [].into_iter().collect(),
        sampler_no_unnormalized_coordinates: [].into_iter().collect(),
        sampler_no_ycbcr_conversion: [].into_iter().collect(),
        sampler_with_images: [].into_iter().collect(),
        stages: ::vulkano::shader::ShaderStages {
            vertex: false,
            tessellation_control: false,
            tessellation_evaluation: false,
            geometry: false,
            fragment: false,
            compute: true,
            raygen: false,
            any_hit: false,
            closest_hit: false,
            miss: false,
            intersection: false,
            callable: false,
        },
        storage_image_atomic: [].into_iter().collect(),
        storage_read: [].into_iter().collect(),
        storage_write: [0u32].into_iter().collect(),
    };
    unsafe {
        Ok(::vulkano::shader::ShaderModule::from_words_with_data(
            device,
            spirv_words,
            ::vulkano::Version {
                major: 1u32,
                minor: 0u32,
                patch: 0u32,
            },
            [&::vulkano::shader::spirv::Capability::Shader],
            [],
            [(
                "main".to_owned(),
                ::vulkano::shader::spirv::ExecutionModel::GLCompute,
                ::vulkano::shader::EntryPointInfo {
                    execution: ::vulkano::shader::ShaderExecution::Compute,
                    descriptor_requirements: (0..param_count as u32)
                        .into_iter()
                        .map(|ix| ((0, ix), req.clone()))
                        .collect(),
                    push_constant_requirements: None,
                    specialization_constant_requirements: Default::default(),
                    input_interface: ::vulkano::shader::ShaderInterface::new_unchecked(vec![]),
                    output_interface: ::vulkano::shader::ShaderInterface::new_unchecked(vec![]),
                },
            )],
        )?)
    }
}
#[allow(non_snake_case)]
#[repr(C)]
pub struct SpecializationConstants {}
#[automatically_derived]
#[allow(non_snake_case)]
impl ::core::fmt::Debug for SpecializationConstants {
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        ::core::fmt::Formatter::write_str(f, "SpecializationConstants")
    }
}
#[automatically_derived]
#[allow(non_snake_case)]
impl ::core::marker::Copy for SpecializationConstants {}
#[automatically_derived]
#[allow(non_snake_case)]
impl ::core::clone::Clone for SpecializationConstants {
    #[inline]
    fn clone(&self) -> SpecializationConstants {
        *self
    }
}
impl Default for SpecializationConstants {
    fn default() -> SpecializationConstants {
        SpecializationConstants {}
    }
}
unsafe impl ::vulkano::shader::SpecializationConstants for SpecializationConstants {
    fn descriptors() -> &'static [::vulkano::shader::SpecializationMapEntry] {
        static DESCRIPTORS: [::vulkano::shader::SpecializationMapEntry; 0usize] = [];
        &DESCRIPTORS
    }
}
pub mod ty {
    #[repr(C)]
    #[allow(non_snake_case)]
    pub struct Data {
        pub data: [u32],
    }
}
