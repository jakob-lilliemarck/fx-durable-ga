use burn::{
    data::dataloader::batcher::Batcher,
    prelude::*,
};

pub const NUM_FEATURES: usize = 8;

#[derive(Clone, Debug)]
pub struct HousingItem {
    pub features: [f32; NUM_FEATURES],
    pub target: f32,
}

pub struct HousingDataset {
    items: Vec<HousingItem>,
}

impl HousingDataset {
    /// Create a synthetic housing dataset that mimics California Housing
    pub fn new() -> Self {
        use rand::{Rng, SeedableRng};
        use rand::rngs::StdRng;
        
        let mut rng = StdRng::seed_from_u64(42);
        let mut items = Vec::new();
        
        // Generate 16640 samples (similar to California Housing train size)
        for _ in 0..16640 {
            let median_income = rng.gen_range(0.5..15.0);
            let house_age = rng.gen_range(1.0..52.0);
            let avg_rooms = rng.gen_range(1.0..40.0);
            let avg_bedrooms = rng.gen_range(0.1..8.0);
            let population = rng.gen_range(3.0..35682.0);
            let avg_occupancy = rng.gen_range(0.7..1243.0);
            let latitude = rng.gen_range(32.5..42.0);
            let longitude = rng.gen_range(-124.4..-114.3);
            
            // Synthetic target based on realistic relationships
            let target = median_income * 0.5 + 
                         (house_age / 52.0) * -0.2 + 
                         (avg_rooms / 10.0) * 0.1 +
                         rng.gen_range(-0.5..0.5); // Add noise
            
            items.push(HousingItem {
                features: [
                    median_income,
                    house_age,
                    avg_rooms,
                    avg_bedrooms,
                    population,
                    avg_occupancy,
                    latitude,
                    longitude,
                ],
                target: target.max(0.1), // Ensure positive
            });
        }
        
        Self { items }
    }
    
    pub fn train(&self) -> Vec<HousingItem> {
        let end_idx = (self.items.len() as f32 * 0.8) as usize;
        self.items[0..end_idx].to_vec()
    }
    
    pub fn validation(&self) -> Vec<HousingItem> {
        let start_idx = (self.items.len() as f32 * 0.8) as usize;
        self.items[start_idx..].to_vec()
    }
}

#[derive(Clone, Debug)]
pub struct HousingBatcher<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

#[derive(Clone, Debug)]
pub struct HousingBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1>,
}

impl<B: Backend> HousingBatcher<B> {
    pub fn new(_device: B::Device) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Batcher<B, HousingItem, HousingBatch<B>> for HousingBatcher<B> {
    fn batch(&self, items: Vec<HousingItem>, device: &B::Device) -> HousingBatch<B> {
        let mut inputs: Vec<Tensor<B, 2>> = Vec::new();
        let mut targets: Vec<Tensor<B, 1>> = Vec::new();

        for item in items.iter() {
            let input_tensor = Tensor::<B, 1>::from_floats(item.features, device).unsqueeze();
            let target_tensor = Tensor::<B, 1>::from_floats([item.target], device);

            inputs.push(input_tensor);
            targets.push(target_tensor);
        }

        let inputs = Tensor::cat(inputs, 0);
        let targets = Tensor::cat(targets, 0);

        HousingBatch { inputs, targets }
    }
}
