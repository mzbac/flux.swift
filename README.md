# FLUXSwift

FLUXSwift is a Swift implementation of the FLUX.1 model family (Schnell, Dev, and Kontext), using mlx-swift for GPU acceleration on Apple Silicon.

## Requirements

- Swift 6.0
- Apple Silicon Mac

## Features

- 🚀 Fast inference on Apple Silicon using MLX
- 📦 Support for quantized models (4-bit, 8-bit)
- 💾 **NEW: Save and load pre-quantized weights for 3-5x faster loading**
- 🎨 Multiple model variants (Schnell, Dev, Kontext)
- 🖼️ Image-to-image generation with Kontext model
- 🎭 LoRA support for fine-tuned models

## Installation

Add FLUX Swift to your project using Swift Package Manager. Add the following dependency to your `Package.swift` file:

```swift
dependencies: [
.package(url: "https://github.com/mzbac/flux.swift.git", from: "0.1.3")
]
```


## Usage

### Available Models

FLUX Swift supports three models:
- **FLUX.1-schnell**: Fast inference model (default, no token required)
- **FLUX.1-dev**: Development model (requires Hugging Face token)
- **FLUX.1-Kontext-dev**: Image-to-image transformation model (requires Hugging Face token)

### Text-to-Image Generation

Here's how to use FLUX Swift programmatically in your Swift projects:

```swift
import FluxSwift
import Hub

// Initialize the FLUX model
let config = FluxConfiguration.flux1Schnell
let loadConfig = LoadConfiguration(float16: true, quantize: false)
let generator = try config.textToImageGenerator(configuration: loadConfig)

// Set up parameters
var params = config.defaultParameters()
params.prompt = "A beautiful sunset over the ocean"
params.width = 768
params.height = 512

// Generate image latents
var denoiser = generator.generateLatents(parameters: params)
var lastXt: MLXArray!
while let xt = denoiser.next() {
    print("Step \(denoiser.i)/\(params.numInferenceSteps)")
    lastXt = xt
}

func unpackLatents(_ latents: MLXArray, height: Int, width: Int) -> MLXArray {
    let reshaped = latents.reshaped(1, height / 16, width / 16, 16, 2, 2)
    let transposed = reshaped.transposed(0, 1, 4, 2, 5, 3)
    return transposed.reshaped(1, height / 16 * 2, width / 16 * 2, 16)
}

// Decode latents to image
let unpackedLatents = unpackLatents(lastXt, height: params.height, width: params.width)
let decoded = generator.decode(xt: unpackedLatents)

// Process and save the image
let imageData = decoded.squeezed()
let raster = (imageData * 255).asType(.uint8)
let image = Image(raster)
try image.save(url: URL(fileURLWithPath: "output.png"))
```

### Image-to-Image Generation with Kontext

FLUX.1-Kontext-dev supports image-to-image transformations:

```swift
import FluxSwift
import Hub
import MLX

// Initialize Kontext model (requires Hugging Face token)
let config = FluxConfiguration.flux1KontextDev
let hub = HubApi(hfToken: "your-hf-token")
let loadConfig = LoadConfiguration(float16: true, quantize: false)
let generator = try config.kontextImageToImageGenerator(hub: hub, configuration: loadConfig)!

// Load and preprocess input image
let inputImage = try Image(url: URL(fileURLWithPath: "input.jpg"))
let normalized = (inputImage.data.asType(.float32) / 255) * 2 - 1

// Set up parameters
var params = config.defaultParameters()
params.prompt = "Transform this into a watercolor painting"
params.width = 1024  // Output dimensions
params.height = 768

// Generate transformed image
var denoiser = generator.generateKontextLatents(
    image: normalized,
    parameters: params
)

var lastXt: MLXArray!
while let xt = denoiser.next() {
    print("Step \(denoiser.i)/\(params.numInferenceSteps)")
    eval(xt)
    lastXt = xt
}

// Decode and save the result
let unpackedLatents = unpackLatents(lastXt, height: params.height, width: params.width)
let decoded = generator.decode(xt: unpackedLatents)
let imageData = decoded.squeezed()
let raster = (imageData * 255).asType(.uint8)
let outputImage = Image(raster)
try outputImage.save(url: URL(fileURLWithPath: "output.png"))
```

These examples demonstrate how to use both text-to-image generation with FLUX.1 Schnell and image-to-image transformation with FLUX.1-Kontext-dev.

## Quantized Weights (New Feature!)

FLUX Swift now supports saving and loading pre-quantized weights, providing significant performance improvements:

### Benefits
- **3-5x faster loading times**
- **50-75% lower peak memory usage**
- **Consistent quantized weights across runs**

### Quick Example

```swift
// Save quantized weights
let flux = try FluxConfiguration.flux1Schnell.textToImageGenerator(
    configuration: LoadConfiguration(quantize: true)
)
try flux.saveQuantizedWeights(to: URL(fileURLWithPath: "./quantized_schnell"))

// Load pre-quantized weights
let quantizedFlux = try FLUX.loadQuantized(
    from: URL(fileURLWithPath: "./quantized_schnell"),
    modelType: "schnell"
)
```

For detailed usage, see [Quantized Weights Usage Guide](docs/quantized-weights-usage.md).

## Configuration

FLUX Swift provides various configuration options:

1. **FluxConfiguration**: Defines the model architecture and file locations
   - `flux1Schnell`: Fast inference model
   - `flux1Dev`: Development model with enhanced quality
   - `flux1KontextDev`: Image-to-image transformation model

2. **LoadConfiguration**: Specifies loading options
   - `float16`: Use half-precision for memory efficiency (default: true)
   - `quantize`: Enable model quantization
   - `loraPath`: Optional path to LoRA weights

3. **EvaluateParameters**: Sets generation parameters
   - `width`, `height`: Output image dimensions (must be multiples of 16)
   - `numInferenceSteps`: Number of denoising steps
   - `guidance`: Guidance scale for prompt adherence
   - `prompt`: Text description for generation

### Important Notes

- **Hugging Face Token**: Required for Dev and Kontext models. Set via `HubApi(hfToken: "your-token")`
- **Resolution**: Kontext automatically selects optimal resolution based on input aspect ratio
- **Memory**: Use `float16: true` and `quantize: true` for lower memory usage

For detailed configuration options, refer to the `FluxConfiguration.swift` file.

## Acknowledgements

I’d like to thank the following projects for inspiring and guiding the development of this work:

- [mflux](https://github.com/filipstrand/mflux) - A MLX port of FLUX
- [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples) - Examples using MLX Swift.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.