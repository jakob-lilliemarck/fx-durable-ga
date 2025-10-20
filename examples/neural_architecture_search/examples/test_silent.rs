use burn::backend::{ndarray::NdArray, Autodiff};
use neural_architecture_search::{training::train_silent, model::RegressionModelConfig};

type Backend = Autodiff<NdArray>;

fn main() {
    println!("Testing silent training...");
    
    let device = Default::default();
    
    // Test with a simple model configuration
    let model_config = RegressionModelConfig::new();
    
    // Train for just 5 epochs to test quickly
    let validation_loss = train_silent::<Backend>(model_config, device, 5);
    
    println!("Silent training completed!");
    println!("Final validation loss: {:.6}", validation_loss);
    
    // Test should complete without showing any TUI
    if validation_loss > 0.0 && validation_loss < f32::MAX {
        println!("✅ Silent training test passed!");
    } else {
        println!("❌ Silent training test failed - invalid loss value");
    }
}