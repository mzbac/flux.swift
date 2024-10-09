# FLUXSwift

FLUXSwift is a Swift implementation of the FLUX.1 model, it uses the mlx-swift for gpu acceleration on Apple Silicon.

## Requirements

- Swift 6.0
- Apple Silicon Mac

## Installation

Add FLUX Swift to your project using Swift Package Manager. Add the following dependency to your `Package.swift` file:

```swift
dependencies: [
.package(url: "https://github.com/mzbac/flux.swift.git", from: "0.1.1")
]
```


## Usage

Here's how to use FLUX Swift programmatically in your Swift projects:

```swift
import FluxSwift

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
    let transposed = reshaped.transposed(0, 3, 1, 4, 2, 5)
    return transposed.reshaped(1, 16, height / 16 * 2, width / 16 * 2)
}

// Decode latents to image
let unpackedLatents = unpackLatents(lastXt, height: params.height, width: params.width)
let decoded = generator.decode(xt: unpackedLatents)

// Process and save the image
let imageData = decoded.squeezed().transposed(1, 2, 0)
let raster = (imageData * 255).asType(.uint8)
let image = Image(raster)
try image.save(url: URL(fileURLWithPath: "output.png"))
```
This example demonstrates how to initialize the FLUX.1 Schnell model, set up generation parameters, generate image latents, decode them into an image, and save the result.

## Configuration

FLUX Swift provides various configuration options:

1. `FluxConfiguration`: Defines the model architecture and file locations.
2. `LoadConfiguration`: Specifies loading options such as data type and quantization.
3. `EvaluateParameters`: Sets parameters for the text-to-image generation process.

For detailed configuration options, refer to the `FluxConfiguration.swift` file.

## TODO

- [ ] Support for FLUX.1 Dev model
- [ ] Integration of LoRA
- [ ] Image-to-image generation

## Acknowledgements

I’d like to thank the following projects for inspiring and guiding the development of this work:

- [mflux](https://github.com/filipstrand/mflux) - A MLX port of FLUX
- [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples) - Examples using MLX Swift.
