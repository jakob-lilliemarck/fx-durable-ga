use burn::backend::{ndarray::NdArray, Autodiff};
use neural_architecture_search::{training::train_silent, model::RegressionModelConfig};

type Backend = Autodiff<NdArray>;

fn main() {
    println!("Testing different architectures to verify real training...");
    
    let device: <Backend as burn::prelude::Backend>::Device = Default::default();
    
    // Test with different hidden sizes
    let configs = vec![
        (32, "Small"),
        (64, "Medium"), 
        (128, "Large"),
        (256, "Very Large"),
    ];
    
    for (hidden_size, name) in configs {
        let mut model_config = RegressionModelConfig::new();
        model_config.hidden_size = hidden_size;
        
        println!("\nTesting {} architecture (hidden_size: {})", name, hidden_size);
        let validation_loss = train_silent::<Backend>(model_config, device.clone(), 3);
        println!("  -> Validation loss: {:.6}", validation_loss);
    }
    
    println!("\n✅ If the losses are different between architectures, training is working!");
    println!("⚠️  If all losses are identical, training might not be working properly.");
}