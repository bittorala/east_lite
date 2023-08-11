<script setup lang="ts">
// This starter template is using Vue 3 <script setup> SFCs
// Check out https://vuejs.org/api/sfc-script-setup.html#script-setup
import ImageInferrer from "./components/ImageInferrer.vue";
</script>

<template>
  <section id="app">
    <nav class="navbar" role="navigation" aria-label="main navigation">
      <div class="navbar-brand" style="width: 100%">
        <div class="navbar-menu">
          <div class="navbar-start">
            <a class="navbar-item"> Home </a>
            <a class="navbar-item"> About </a>
          </div>
          <div class="navbar-end">
            <span class="navbar-item title is-4">EAST Lite</span>
          </div>
        </div>
      </div>
    </nav>
    <ImageInferrer />
  </section>
</template>

<style lang="scss">
@import "./assets/main.scss";
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  height: 98vh;
}

nav {
  padding: 16px;
  background-color: darkslategrey !important;
}

.navbar {
  span,
  a {
    color: white;
  }
}
</style>
