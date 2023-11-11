
// List of image filenames
var imageList = [];
var textList = []; 
var currentIndex = 0;

// Function to update the displayed image
function updateImage() {
    document.getElementById("image-viewer").src = imageList[currentIndex];
    document.getElementById("text-viewer").src = textList[currentIndex];
}

// Function to handle key presses
function handleKeyPress(event) {
    if (event.key === "ArrowLeft" && currentIndex > 0) {
        currentIndex--;
        updateImage();
    } else if (event.key === "ArrowRight" && currentIndex < imageList.length - 1) {
        currentIndex++;
        updateImage();
    } else if (event.key === "ArrowLeft" && currentIndex <= 0) {
        currentIndex = imageList.length - 1; 
        updateImage();
    } else if (event.key === "ArrowRight" && currentIndex >= imageList.length - 1) {
        currentIndex = 0; 
        updateImage();
    }
}

var path = "../src/rend/pic/Cynthia_full/" + sign + "/"; 
var text_src = "../src/hs/"

// Populate the image list with filenames
// You should replace "path/to/images/" with the actual path to your image directory
for (var i = 0; i < filenum; i++) {
    var filename = path + sign + "_" + i.toString().padStart(6, '0') + ".jpg";
    imageList.push(filename);
}

for (var i = 0; i < preds.length; i++) {
    var filename = text_src + preds[i] + ".png";
    textList.push(filename);
}

// Set up initial image
updateImage();

// Add event listener for key presses
document.addEventListener("keydown", handleKeyPress);