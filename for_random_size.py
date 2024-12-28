device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_samples = primary_samples + backup_samples

PADDING = KERNEL_SIZE // 2
image_path = image_file_name
original_image = Image.open(image_path)
original_image = original_image.resize((2281, 1509))  # Resize to required
original_image = original_image.convert('RGB')
original_array = np.array(original_image)  
original_array = original_array / 255.0
height, width, _ = original_array.shape  

image_array = original_array
target_tensor = torch.tensor(image_array, dtype=torch.float32, device=device)
coords = np.random.randint(0, [width, height], size=(num_samples, 2))  # Ensure correct coords generation
random_pixel_means = torch.tensor(coords, device=device)
pixels = [image_array[coord[1], coord[0]] for coord in coords]  # Corrected indexing for image_array
pixels_np = np.array(pixels)
random_pixels = torch.tensor(pixels_np, device=device)

# Assuming give_required_data function uses the correct image size (2281x1509)
colour_values, pixel_coords = give_required_data(coords, (height, width))

colour_values = torch.logit(colour_values)
pixel_coords = torch.atanh(pixel_coords)

scale_values = torch.logit(torch.rand(num_samples, 2, device=device))
rotation_values = torch.atanh(2 * torch.rand(num_samples, 1, device=device) - 1)
alpha_values = torch.logit(torch.rand(num_samples, 1, device=device))
W_values = torch.cat([scale_values, rotation_values, alpha_values, colour_values, pixel_coords], dim=1)
