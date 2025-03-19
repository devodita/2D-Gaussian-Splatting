# Save final 2D Gaussian Splat and generated array
final_generated_array = g_tensor_batch.cpu().detach().numpy()

# Save the 2D Gaussian Splat as an image
final_img = Image.fromarray((final_generated_array * 255).astype(np.uint8))
final_img.save(os.path.join(directory, "final_2d_gaussian_splat.jpg"))

# Save the generated array as a .npy file
np.save(os.path.join(directory, "final_generated_array.npy"), final_generated_array)

print("Final 2D Gaussian Splat and generated array saved successfully.")
