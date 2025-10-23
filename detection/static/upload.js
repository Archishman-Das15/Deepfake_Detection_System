// detection/static/upload.js

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('uploadForm');
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');

    form.addEventListener('submit', function(e) {
        e.preventDefault(); // prevent normal submission

        const fileInput = form.querySelector('input[name="video"]');
        const file = fileInput.files[0];
        if (!file) return alert('Please select a video file.');

        progressContainer.style.display = 'block';
        progressBar.style.width = '0%';
        progressBar.textContent = '0%';

        let percent = 0;
        const interval = setInterval(() => {
            percent += Math.floor(Math.random() * 10) + 1; // increment by random 1-10%
            if (percent >= 100) {
                percent = 100;
                clearInterval(interval);

                progressBar.textContent = 'Processing...';
                setTimeout(() => form.submit(), 500); // submit form after fake progress
            }
            progressBar.style.width = percent + '%';
            progressBar.textContent = percent + '%';
        }, 200);
    });
});