const webcam = document.getElementById("webcam");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let model;
let modelURL;
let imageDims;
let canvasDims;
let modelChange;

function fetchInputImage() {
  return tf.tidy(() => {
    const webcamImage = tf.browser.fromPixels(webcam);

    const batchedImage = webcamImage.toFloat().expandDims();

    const resizedImage = tf.image.resizeBilinear(batchedImage, imageDims, true);

    const clippedImage = tf.clipByValue(resizedImage, 0.0, 255.0);

    const reversedImage = tf.reverse(clippedImage, 2);

    return reversedImage;
  });
}


function predictSaliency() {
  return tf.tidy(() => {
    const modelOutput = model.predict(fetchInputImage());

    const resizedOutput = tf.image.resizeBilinear(modelOutput, canvasDims, true);

    const clippedOutput = tf.clipByValue(resizedOutput, 0.0, 255.0);

    return clippedOutput.squeeze();
  });
}


async function runModel() {
  showLoadingScreen();

  model = await tf.loadGraphModel(modelURL);

  tf.tidy(() => model.predict(fetchInputImage())); 

  modelChange = false;

  while (!modelChange) {
    const saliencyMap = predictSaliency();

    await tf.browser.toPixels(saliencyMap, canvas);

    saliencyMap.dispose();

    await tf.nextFrame();
  }

  model.dispose();

  runModel();
}


function showLoadingScreen() {
  ctx.fillStyle = "black";
  ctx.textAlign = "center";
  ctx.font = "1.7em Alegreya Sans SC", "1.7em sans-serif";
  ctx.fillText("loading model...", canvas.width / 2, canvas.height / 2);
}


async function setupWebcam() {
  if (navigator.mediaDevices.getUserMedia) {
    const stream = await navigator.mediaDevices.getUserMedia({
      "audio": false,
      "video": {
        width: {
          min: 640,
          max: 640
        },
        height: {
          min: 480,
          max: 480
        }
      }
    });

    webcam.srcObject = stream;

    return new Promise((resolve) => {
      webcam.onloadedmetadata = () => {
        webcam.width = stream.getVideoTracks()[0].getSettings().width;
        webcam.height = stream.getVideoTracks()[0].getSettings().height;
        canvas.width = stream.getVideoTracks()[0].getSettings().width;
        canvas.height = stream.getVideoTracks()[0].getSettings().height;

        canvasDims = [canvas.height, canvas.width];

        resolve(webcam);
      };
    });
  }
}


async function app() {
  modelURL = "https://storage.googleapis.com/msi-net/model/very_low/model.json";
  imageDims = [48, 64];

  document.getElementById("very_low").addEventListener("click", () => {
    modelURL = "https://storage.googleapis.com/msi-net/model/very_low/model.json";
    imageDims = [48, 64];
    modelChange = true;
  });

  document.getElementById("low").addEventListener("click", () => {
    modelURL = "https://github.com/videetnimsarkar21/Visual_Saliency/blob/main/SalNET.json";
    imageDims = [72, 96];
    modelChange = true;
  });

  const noWebcamError = document.getElementById("error");

  try {
    await setupWebcam();
    noWebcamError.style.display = "none";
  } catch (DOMException) {
    noWebcamError.style.visibility = "visible";
    return;
  }

  runModel();
}

app();