import axios from 'axios';

const http = axios.create({
  baseURL: 'http://localhost:8000',
  headers: {
    "Content-type": "application/json"
  }
});

function upload(file: any, onUploadProgress: any) {
  let formData = new FormData();
  formData.append("file", file);
  return http.post("/image", formData, {
    headers: {
    "Content-Type": "multipart/form-data"
    },
    onUploadProgress
  });
}

export { upload };
