from flask import Flask, render_template, request
import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
from skimage import transform
from skimage.io import imread
from sklearn.cluster import MiniBatchKMeans
from collections import Counter
import os

# =========================================================
app = Flask(__name__)
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

# =========================================================
# The sample image displayed on the homepage by defailt
SAMPLE_FP = './static/img/sample.jpg'
# The most recent image uploaded by the user
RECENT_OPENED = None
# Default values for the form
DEFAULT_VALUES = {
        'num_results': 10,
        'delta': 100,
        'brightness': 100,
    }

# =========================================================
@app.route('/')
def index():
    """Open the homepage and display the sample image with the top 10 colors."""
    global SAMPLE_FP, DEFAULT_VALUES

    # Extract the top 10 colors from the sample image
    sorted_colors = get_main_colors(
        img_url = SAMPLE_FP,
        color_no = DEFAULT_VALUES['num_results'], 
        delta = DEFAULT_VALUES['delta'], 
        brightness = DEFAULT_VALUES['brightness'],
    )
    # Get the beautiful logo colors
    logo_color = get_logo_colors(sorted_colors)
    
    return render_template('index.html', img=SAMPLE_FP, colors_top10=sorted_colors, logo_color=logo_color, form_values=DEFAULT_VALUES, errors=None)

# =========================================================
@app.route('/color_extract', methods=['GET', 'POST'])
def color_extract():
    """
    Upload an Image & Extract Colors:
    1. When 1st accessing the site, a sample image is displayed by default, showing the top 10 colors.
    2. Users can upload their own images, adjust parameters, and submit to view the results.
    3. Users can modify individual parameters, and upon submission, the analysis and page rendering will update based on the latest settings.
    """
    global SAMPLE_FP, RECENT_OPENED, DEFAULT_VALUES

    # Initialize the form values and errors
    form_values= DEFAULT_VALUES
    errors = {}

    if request.method == 'POST':
        # Get image file path from user.
        try:
            img_file = request.files['imgFile']
            if img_file:
                # When user uploads an image, save the image path to RECENT_OPENED.
                if img_file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    fp = os.path.join('./static/img', img_file.filename)
                    img_file.save(fp)
                    RECENT_OPENED = fp
                # If the selected file is not an image, load the sample image instead.
                else:
                    fp = SAMPLE_FP
                    errors['imgFile'] = '** Invalid file format. Please upload an image file (JPG/JPEG/PNG/GIF).'
            # Ensure the previously uploaded image remains in use when user modified any parameter and re-submitted
            elif RECENT_OPENED:
                fp = RECENT_OPENED
            # If there's no previously updaloaded image, Load the sample image when user refreshed the page.
            else:
                fp = SAMPLE_FP
            img_url = f'/{fp}'
        except ValueError:
            errors['imgFile'] = 'Invalid file format.'

        # Get 'num_results' set by the user.
        try:
            num_results = int(request.form.get('num_results'))
        except ValueError:
            errors['num_results'] = 'Invalid number format for colors.'

        # Get 'delta' and 'brightness' set by the user.
        delta = int(request.form.get('delta'))
        brightness = int(request.form.get('brightness'))

        # If there are errors, return the page with the errors.
        if errors: 
            return render_template('index.html', img=img_url, form_values=form_values, errors=errors)
        # If there are no errors, proceed to extract the colors and return the results.
        else: 
            form_values.update({
                "num_results": num_results,
                "delta": delta,
                "brightness": brightness,
                "file_name": os.path.basename(img_url),
            })
            sorted_colors = get_main_colors(
                fp, 
                color_no=num_results, 
                delta=delta, 
                brightness=brightness,
                )
            logo_color = get_logo_colors(sorted_colors)

            return render_template('index.html', img=img_url, colors_top10=sorted_colors, logo_color=logo_color, form_values=form_values, errors=errors)

# =========================================================
def get_main_colors(img_url, color_no, delta, brightness):
    """ 
    Method to extract the colors of image, return the result. 
    Adjust the brightness and saturation accordingly.
    :param brightness_factor: 0.0-1.0
    :param suturation_factor: 0.0-1.0
    :return: sorted_colors
    """
    img = imread(img_url)

    # Resize the image to reduce the computation time
    img = transform.rescale(img, [0.6, 0.6, 1]) 
    # Convert the image from RGB to HSV
    img = rgb2hsv(img) 

    # Flatten the image to prepare it for KMeans clustering analysis
    h, w, d = img.shape
    img = np.reshape(img, (h * w, d))

    # Perform KMeans clustering
    kmeans = MiniBatchKMeans(n_clusters=color_no, random_state=0)
    kmeans.fit(img)

    # Retrieve the cluster centers and labels
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Count the number of pixels in each cluster
    label_counts = Counter(labels)
    total_pixels = len(img)

    # Adjust the saturation and brightness of the colors when the user modifies the parameters
    if delta != 100 or brightness != 100:
        cluster_centers[:, 1] = np.clip(cluster_centers[:, 1] * (delta / 100), 0, 1)
        cluster_centers[:, 2] = np.clip(cluster_centers[:, 2] * (brightness / 100), 0, 1)

    # Convert HSV colors to RGB colors in 1 batch operation
    rgb_colors_array = (hsv2rgb(cluster_centers) * 255).astype(int) 
    # Convert the RGB colors to a list of dictionaries, Sort the colors by the number of pixels in each cluster
    rgb_colors = [
        {
            'color': tuple(rgb_colors_array[label]),
            'percentage': round((count / total_pixels) * 100, 2),
            'no': i + 1,
        }
        for i, (label, count) in enumerate(sorted(label_counts.items(), key=lambda x: x[1], reverse=True))
    ]
    
    # Return the sorted colors
    return rgb_colors

# =========================================================
def get_logo_colors(colors, threshold=200):
    """
    <JUST FOR FUN>
    Retrieve 2 colors for the page logo in a linear gradient format:
    - **Start color**: Use the top-ranked color, darkened by 50 units if it is too light ( > threshold ).
    - **End color**: Use the lightest color, determined by the highest brightness value.
    :param colors: A list of colors, as returned by the `get_main_colors` method.
    :param threshold: Brightness threshold, default=200.
    :return: A list containing [start_color, end_color].
    """
    r, g, b = colors[0]['color']
    if 0.2126 * r + 0.7152 * g + 0.0722 * b > threshold:
        start_color = tuple(map(lambda x: x - 50, colors[0]['color']))
    else:
        start_color = colors[0]['color']

    def end_color(color):
        r,g,b = map(int, color)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    return [start_color, max([item['color'] for item in colors if 'color' in item], key=end_color)]

# =========================================================
if __name__ == '__main__':
    app.run(debug=True)