Task:
Build a website that allows users to upload images and analyze the main colors, displaying them sorted in descending order.

Demo:
https://www.coolphptools.com/color_extract#demo

Task Breakdown:

1. Set up the backend using Flask to handle image uploads.
2. Use KMeans clustering to extract the specified number of cluster center colors from the uploaded image and display the colors sorted in descending order based on their proportion in the image.
3. Configurable parameters: Image file upload, number of colors, saturation adjustment, and brightness adjustment.
4. Output: The image, the specified number of top-ranked colors, their RGB values, and their respective proportions.

The Most Challenging Parts:

- Representing colors in different formats (e.g., RGB, HSV, HSL, etc.)
- Converting colors in images into quantifiable data and analyzing them
- Implementing the KMean clustering logic

The Simple Parts:

- Setting up the program framework using Flask, HTML, and CSS
- Displaying the processed data from Flask on the HTML webpage.
