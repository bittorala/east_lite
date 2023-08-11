<template>
  <section>
    <div class="uploadinput block">
      <input type="file" id="myfile" ref="file" accept="image/*" @input="onFileInput" />
    </div>
    <div class="content tile is-ancestor">
      <div class="card block tile is-child is-8">
        <div id="image-panel" class="card-content">
          <canvas
            id="canvas"
            :width="resizedWidth"
            :height="resizedHeight"
            :style="`background: url(${previewImage})`"
          />
        </div>
        <div class="hero is-small is-primary">
          <div class="hero-body">
            <p clas="title">Loaded</p>
          </div>
        </div>
      </div>
      <div
        v-if="inferring"
        class="tile card is-child is-4 progress-bar"
        style="overflow-y: scroll"
      >
        <progress class="progress is-medium is-dark" max="100"></progress>
      </div>
      <div v-else class="tile card is-child is-4" style="overflow-y: scroll">
        <span class="card-content" v-for="(box, index) in boxesFlat" :key="index">
          {{ JSON.stringify(box) }}
        </span>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref } from "vue";
import type { Ref } from "vue";
import axios from "axios";

const file: Ref<HTMLInputElement | null> = ref(null);
const previewImage: Ref<string | undefined> = ref(undefined);
const boxesFlat: Ref<Array<Array<number>>> = ref([]);

const MAX_SIZE: number = Math.max(window.innerWidth / 2, window.innerHeight * 0.7);

const resizedHeight: Ref<number> = ref(MAX_SIZE);
const resizedWidth: Ref<number> = ref(MAX_SIZE);
const inferring: Ref<boolean> = ref(false);

async function onFileInput() {
  if (file.value && file.value.files![0]) {
    previewImage.value = URL.createObjectURL(file.value.files![0]);
  }

  const img = new Image();
  img.src = previewImage.value!;
  await new Promise((resolve) => {
    img.onload = () => resolve();
  });
  const c = document.getElementById("canvas") as HTMLCanvasElement;
  const ctx = c!.getContext("2d")!;
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.drawImage(img, 0, 0, resizedWidth.value, resizedHeight.value);

  const [width, height] = [img.width, img.height];
  const factor = Math.max(height, width) / MAX_SIZE;
  [resizedWidth.value, resizedHeight.value] = [width / factor, height / factor];
  inferText(factor);
}

async function inferText(factor: number) {
  const formData = new FormData();
  formData.append("image", file.value!.files![0]);
  inferring.value = true;
  const response = await axios.post("http://localhost:8000/image/", formData, {
    headers: {
      Accept: "application/json",
      "Content-Type": "multipart/form-data",
    },
  });
  inferring.value = false;
  boxesFlat.value = response.data.array.map((b: any) =>
    b.flatMap((coords: any) => coords)
  );
  drawText(factor);
}

function drawText(factor: number) {
  const c = document.getElementById("canvas") as HTMLCanvasElement;
  const ctx = c!.getContext("2d")!;
  ctx.clearRect(0, 0, c.width, c.height);

  for (const box of boxesFlat.value) {
    for (let i = 0; i < 4; ++i) {
      const [x0, y0] = box.slice(i * 2, (i + 1) * 2);
      const [x1, y1] = box.slice(((i + 1) * 2) % 8, (((i + 2) * 2 + 7) % 8) + 1);
      ctx.beginPath();
      ctx.moveTo(x0 / factor, y0 / factor);
      ctx.lineTo(x1 / factor, y1 / factor);
      ctx.lineWidth = 2;
      ctx.strokeStyle = "#00ffff";
      ctx.stroke();
    }
  }
}
</script>

<style scoped>
canvas {
  background-size: contain !important;
  background-repeat: no-repeat !important;
}

.progress-bar {
  display: flex;
  justify-content: center;
  align-items: center;
}
</style>
