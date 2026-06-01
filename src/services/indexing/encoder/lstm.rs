use burn::nn;
use burn::prelude::*;
use serde::Deserialize;
use serde::Serialize;
use tracing::instrument;

/// Common configuration for sequential autoencoders.
///
/// * `input_size` – dimensionality of each timestep.
/// * `hidden_size` – LSTM hidden/state size.
/// * `latent_size` – dimensionality of the compressed embedding.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct AutoencoderConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub latent_size: usize,
}

/// Output returned by autoencoder forward passes.
#[derive(Debug)]
pub struct AutoencoderOutput<B: Backend> {
    pub reconstruction: Tensor<B, 3>,
    pub latent: Tensor<B, 2>,
}

/// Trait implemented by all sequential autoencoder models.
pub trait AutoencoderModel<B: Backend>: Module<B> + Sized {
    fn new(device: &B::Device, config: AutoencoderConfig) -> Self;

    fn forward(&self, input: Tensor<B, 3>) -> AutoencoderOutput<B>;
}

/// Autoencoder backed by an encoder/decoder LSTM pair.
#[derive(Module, Debug)]
pub struct LstmAutoencoder<B: Backend> {
    /// Encoder consumes the raw sequence and returns its final state.
    encoder: nn::Lstm<B>,
    /// Decoder unrolls latent vectors into reconstructed timesteps.
    decoder: nn::Lstm<B>,
    latent_layer: nn::Linear<B>,
    latent_to_hidden: nn::Linear<B>,
    latent_to_cell: nn::Linear<B>,
    decoder_out: nn::Linear<B>,
    #[module(ignore)]
    input_size: usize,
    #[module(ignore)]
    hidden_size: usize,
    #[module(ignore)]
    latent_size: usize,
}

impl<B: Backend> AutoencoderModel<B> for LstmAutoencoder<B> {
    /// Build a new encoder/decoder pair from the provided dimensions.
    #[instrument(level = "debug", skip(device))]
    fn new(device: &B::Device, config: AutoencoderConfig) -> Self {
        let AutoencoderConfig {
            input_size,
            hidden_size,
            latent_size,
        } = config;

        Self {
            encoder: nn::LstmConfig::new(input_size, hidden_size, true)
                .with_batch_first(true)
                .init(device),
            decoder: nn::LstmConfig::new(latent_size, hidden_size, true)
                .with_batch_first(true)
                .init(device),
            latent_layer: nn::LinearConfig::new(hidden_size, latent_size).init(device),
            latent_to_hidden: nn::LinearConfig::new(latent_size, hidden_size).init(device),
            latent_to_cell: nn::LinearConfig::new(latent_size, hidden_size).init(device),
            decoder_out: nn::LinearConfig::new(hidden_size, input_size).init(device),
            input_size,
            hidden_size,
            latent_size,
        }
    }

    /// Encode the sequence, project to latent, and decode back to input space.
    #[instrument(level = "debug", skip(self, input))]
    fn forward(&self, input: Tensor<B, 3>) -> AutoencoderOutput<B> {
        let [batch_size, seq_len, input_size] = input.dims();
        debug_assert_eq!(input_size, self.input_size);

        if seq_len == 0 {
            // Nothing to encode, so emit empty reconstruction and zero latent.
            let reconstruction =
                Tensor::<B, 3>::zeros([batch_size, 0, self.input_size], &input.device());
            let latent = Tensor::<B, 2>::zeros([batch_size, self.latent_size], &input.device());
            return AutoencoderOutput {
                reconstruction,
                latent,
            };
        }

        // Run the encoder across the entire sequence and grab the final state.
        let (_encoder_outputs, encoder_state) = self.encoder.forward(input, None);
        let latent = self
            .latent_layer
            .forward(encoder_state.hidden.clone())
            .tanh();

        // Map latent vector into initial hidden/cell states for the decoder.
        let decoder_hidden = self.latent_to_hidden.forward(latent.clone()).tanh();
        let decoder_cell = self.latent_to_cell.forward(latent.clone()).tanh();
        let decoder_state = nn::LstmState::new(decoder_cell, decoder_hidden);

        // Feed the decoder a constant latent token at each step.
        let repeated_latents: Vec<_> = (0..seq_len).map(|_| latent.clone()).collect();
        let decoder_input = Tensor::stack(repeated_latents, 1);
        let (decoder_outputs, _) = self.decoder.forward(decoder_input, Some(decoder_state));

        // Project decoder outputs back to the original feature dimension.
        let decoder_outputs = decoder_outputs.reshape([batch_size * seq_len, self.hidden_size]);
        let reconstructed_flat = self.decoder_out.forward(decoder_outputs);
        let reconstruction = reconstructed_flat.reshape([batch_size, seq_len, self.input_size]);

        AutoencoderOutput {
            reconstruction,
            latent,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    pub use burn_wgpu::Wgpu;

    #[test]
    /// Confirms reconstruction/latent shapes stay consistent for varying lengths.
    fn lstm_autoencoder_handles_variable_lengths() {
        let device = Default::default();
        let config = AutoencoderConfig {
            input_size: 4,
            hidden_size: 8,
            latent_size: 3,
        };

        let model = LstmAutoencoder::<Wgpu>::new(&device, config);

        for seq_len in [1, 5, 9] {
            let input = Tensor::<Wgpu, 3>::ones([2, seq_len, config.input_size], &device);
            let output = model.forward(input);

            assert_eq!(
                output.reconstruction.dims(),
                [2, seq_len, config.input_size]
            );
            assert_eq!(output.latent.dims(), [2, config.latent_size]);
        }
    }
}
