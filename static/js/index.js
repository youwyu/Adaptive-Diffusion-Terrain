// Dependency-free page script: lazy video loading, the diffusion-step slider, and a two-part video player.
(function () {
  "use strict";

  // Videos load only when they come near the viewport, and pause when they leave it.
  function loadVideo(video) {
    if (video.dataset.loaded) return;
    video.dataset.loaded = "1";
    if (video.classList.contains("sequence-video")) {
      setupSequence(video);
    } else {
      video.querySelectorAll("source[data-src]").forEach(function (s) { s.src = s.dataset.src; });
      video.preload = "auto";
      video.load();
    }
  }

  function playIfAllowed(video) {
    if (!("autoplay" in video.dataset)) return;
    var p = video.play();
    if (p && p.catch) p.catch(function () {});
  }

  // Plays a list of files back to back (lossless split of a clip that exceeds GitHub's file-size limit).
  function setupSequence(video) {
    var parts = video.dataset.parts.trim().split(/\s+/);
    var i = 0;
    var next = document.createElement("link");
    next.rel = "preload"; next.as = "video";
    video.src = parts[0];
    video.preload = "auto";
    video.addEventListener("playing", function () {
      if (i + 1 < parts.length && !next.href) { next.href = parts[i + 1]; document.head.appendChild(next); }
    });
    video.addEventListener("ended", function () {
      i = (i + 1) % parts.length;
      video.src = parts[i];
      next.removeAttribute("href");
      var p = video.play();
      if (p && p.catch) p.catch(function () {});
    });
  }

  var videos = document.querySelectorAll("video.lazy-video");
  if ("IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        var v = e.target;
        if (e.isIntersecting) { loadVideo(v); playIfAllowed(v); }
        else if (!v.paused) { v.pause(); }
      });
    }, { rootMargin: "200px 0px" });
    videos.forEach(function (v) { io.observe(v); });
  } else {
    videos.forEach(function (v) { loadVideo(v); playIfAllowed(v); });
  }

  // Diffusion-step slider: frames are fetched once the panel is near the viewport.
  var slider = document.getElementById("step-interpolation-slider");
  var wrapper = document.getElementById("step-interpolation-image-wrapper");
  if (slider && wrapper) {
    var frames = [];
    var img = wrapper.querySelector("img");
    function frameSrc(i) { return "./static/interpolation/step/" + String(i).padStart(6, "0") + ".jpg"; }
    function preload() {
      if (frames.length) return;
      for (var i = 1; i <= 99; i++) { frames[i] = new Image(); frames[i].src = frameSrc(i); }
    }
    slider.addEventListener("input", function () { preload(); img.src = frameSrc(this.value); });
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (entries, obs) {
        if (entries[0].isIntersecting) { preload(); obs.disconnect(); }
      }, { rootMargin: "300px 0px" }).observe(wrapper);
    } else {
      preload();
    }
  }
})();
