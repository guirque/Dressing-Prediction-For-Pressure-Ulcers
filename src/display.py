import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import colormaps

# Getting Color Pallete (remember to map it with dressings)
# https://matplotlib.org/stable/gallery/color/individual_colors_from_cmap.html
colors = colormaps.get_cmap('tab10').colors

def display_img_boxes(pil_img, labels, class_map):
    """
    Displays image with corresponding bounding boxes.

    # Arguments

    - *pil_img*: pil image to display.
    - *labels*: numpy matrix, with each row containing [label, x, y, w, h], with numbers normalized relative to image size.
    - *class_map*: list of class names (strings). The index represents the class id. 
    """

    # Create subplots
    figure, axes = plt.subplots()
    axes.add_image(plt.imshow(pil_img))

    # For each bounding box
    for label in labels:
        img_w, img_h = (pil_img.width, pil_img.height) # numpy shape will return [c, h, w]

        # https://docs.ultralytics.com/pt/datasets/detect/#ultralytics-yolo-format
        # values are normalized, so we need to "unnormalize" it
        # normalization: x <- px_x_center/img_width, w <- px_width/img_width,  y <- px_y_center/img_height, h <- px_height / img_height
        
        # boxes must be centralized! Matplotlib's Rectangle isn't like that by default 
        # (coordinates are xy are treated as a starting point instead).
        # to centralize it, we'll: 
        # - On the x axis, we'll move it half width to the left
        # - On the y axis, we'll move it half height downwards

        # Label ------------
        class_label = class_map[int(label[0])]
        class_color = colors[int(label[0] % len(colors))]

        # Width, Height ----
        w, h = (label[3]*img_w, label[4]*img_h)

        # Coordinates ------
        unnormalized_coordinates = (label[1]*img_w, label[2]*img_h)
        centralized_coordinates = (unnormalized_coordinates[0] - w/2, unnormalized_coordinates[1] - h/2)
        
        # Adding Rectangle and Corresponding Text =
        axes.add_patch(Rectangle(centralized_coordinates, w, h, fill=False, edgecolor=class_color, lw=2))
        plt.text(centralized_coordinates[0]+2, centralized_coordinates[1]-2, f'{class_label}', color='black', weight='bold', backgroundcolor=(1, 1, 1, 0.2))

    plt.show()

    # Other References
    # https://brandonrohrer.com/matplotlib_patches.html
    # https://matplotlib.org/stable/gallery/images_contours_and_fields/image_demo.html
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html
    # https://matplotlib.org/stable/users/explain/colors/colormaps.html