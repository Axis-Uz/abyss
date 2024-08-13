let uploadButton = document.getElementById("upload-button");
let chosenImage = document.getElementById("chosen-image");
let fileName = document.getElementById("file-name");
let container = document.getElementById("container");
let error = document.getElementById("error");
let imageDisplay = document.getElementById("image-display");

const fileHandler = (file, name, type) => {
  if (type.split("/")[0] !== "image" && type.split("/")[0] !== "video") {
    //File Type Error
    error.innerText = "Please upload an image or video file";
    return false;
  }
  if (file.size >= 20 * 1000 * 1000) {
    error.innerText = "Pleas upload a file of size less than 20MB";
    return false;
  }
  error.innerText = "";
  let reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onloadend = () => {
    //image and file name
    if (type.split("/")[0] === "image") {
      let imageContainer = document.createElement("figure");
      let img = document.createElement("img");
      img.src = reader.result;
      imageContainer.appendChild(img);
      imageDisplay.appendChild(imageContainer);
    } else if (type.split("/")[0] === "video") {
      let imageContainer = document.createElement("div");
      let img = document.createElement("video");
      img.src = reader.result;
      imageContainer.appendChild(img);
      imageDisplay.appendChild(imageContainer);
    }
    let sumbitButton = document.createElement("input");
    sumbitButton.addEventListener("click", () => {
      let loader = document.createElement("div");
      loader.classList.add("loader");
      container.appendChild(loader);
      imageDisplay.style.display = "none";
      error.innerHTML = "It will take some time ...";
    });
    sumbitButton.classList.add(
      "block",
      "relative",
      "text-gray-100",
      "text-center",
      "text-lg",
      "capitalize",
      "mt-2",
      "mx-auto",
      "py-2",
      "px-10",
      "w-1/2",
      "rounded-md",
      "shadow-md",
      "cursor-pointer",
      "hvr-grow",
      "bg-violet-700",
      "hover:bg-indigo-600",
      "focus:outline-none",
      "border-0",
      "rounded"
    );
    sumbitButton.type = "submit";
    imageDisplay.appendChild(sumbitButton);
  };
};

//Upload Button
uploadButton.addEventListener("change", () => {
  imageDisplay.innerHTML = "";
  Array.from(uploadButton.files).forEach((file) => {
    fileHandler(file, file.name, file.type);
  });
});

container.addEventListener(
  "dragenter",
  (e) => {
    e.preventDefault();
    e.stopPropagation();
    container.classList.add("active");
  },
  false
);

container.addEventListener(
  "dragleave",
  (e) => {
    e.preventDefault();
    e.stopPropagation();
    container.classList.remove("active");
  },
  false
);

container.addEventListener(
  "dragover",
  (e) => {
    e.preventDefault();
    e.stopPropagation();
    container.classList.add("active");
  },
  false
);

container.addEventListener(
  "drop",
  (e) => {
    e.preventDefault();
    e.stopPropagation();
    container.classList.remove("active");
    let draggedData = e.dataTransfer;
    let files = draggedData.files;
    imageDisplay.innerHTML = "";
    Array.from(files).forEach((file) => {
      fileHandler(file, file.name, file.type);
    });
  },
  false
);

window.onload = () => {
  error.innerText = "";
};
