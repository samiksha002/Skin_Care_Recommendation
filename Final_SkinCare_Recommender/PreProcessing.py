import cv2
import os
from glob import glob

class CLAHEPreprocessor:
    def __init__(self, input_folder, output_folder, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Initializes the CLAHEPreprocessor with input and output folder paths,
        and CLAHE parameters.
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        os.makedirs(self.output_folder, exist_ok=True)  # Create output folder if it doesn't exist
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def apply_clahe(self, img_path):
        """
        Reads an image, applies CLAHE to the L-channel of the LAB color space,
        and returns the enhanced image.
        """
        img = cv2.imread(img_path)  # Read the image in color (BGR)
        if img is None:
            return None
        
        # Convert image to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel (lightness)
        l = self.clahe.apply(l)
        
        # Merge channels and convert back to BGR color space
        enhanced_lab = cv2.merge((l, a, b))
        enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced_img
    
    def process_images(self, file_extension='*.jpg'):
        """
        Processes all images in the input folder with the specified file extension,
        applies CLAHE, and saves the enhanced images to the output folder.
        """
        image_paths = glob(os.path.join(self.input_folder, file_extension))
        for img_path in image_paths:
            enhanced_img = self.apply_clahe(img_path)
            if enhanced_img is not None:
                output_path = os.path.join(self.output_folder, os.path.basename(img_path))
                cv2.imwrite(output_path, enhanced_img)  # Save the processed image
        print(f"CLAHE preprocessing completed. Processed images saved in {self.output_folder}.")


if __name__ == "__main__":
    # Define input and output folders
    input_folder = r'C:\Users\Samiksha Bhatia\Acne_gpu\myvenv\Data\valid\images'
    output_folder = r'C:\Users\Samiksha Bhatia\Acne_gpu\myvenv\Data3\valid\images'
    
    # Create an instance of the preprocessor and process images
    preprocessor = CLAHEPreprocessor(input_folder, output_folder)
    preprocessor.process_images()
