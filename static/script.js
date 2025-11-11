class PlantHealthDetector {
    constructor() {
        this.initializeElements();
        this.bindEvents();
    }

    initializeElements() {
        this.uploadArea = document.getElementById('uploadArea');
        this.imageInput = document.getElementById('imageInput');
        this.imagePreview = document.getElementById('imagePreview');
        this.previewImg = document.getElementById('previewImg');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.resultsSection = document.getElementById('resultsSection');
        this.loadingSection = document.getElementById('loadingSection');
        this.confidenceScore = document.getElementById('confidenceScore');
        this.diseaseName = document.getElementById('diseaseName');
        this.diseaseDescription = document.getElementById('diseaseDescription');
        this.cureSteps = document.getElementById('cureSteps');
    }

    bindEvents() {
        this.uploadArea.addEventListener('click', () => this.imageInput.click());
        this.uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        this.uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        this.uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        this.imageInput.addEventListener('change', this.handleFileSelect.bind(this));
        this.analyzeBtn.addEventListener('click', this.analyzeImage.bind(this));
    }

    handleDragOver(e) {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.processFile(file);
        }
    }

    processFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file.');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImg.src = e.target.result;
            this.uploadArea.style.display = 'none';
            this.imagePreview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    async analyzeImage() {
        this.showLoading();
        
        try {
            // Get the image data from the preview
            const imageData = this.previewImg.src;
            
            // Call the actual Flask API
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            this.displayResults(result);
        } catch (error) {
            console.error('Analysis failed:', error);
            alert('Analysis failed. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    displayResults(result) {
        this.confidenceScore.textContent = `${result.confidence}%`;
        this.diseaseName.textContent = result.disease;
        this.diseaseDescription.textContent = result.description;
        
        // Handle cure steps (they come as an array)
        const cureList = Array.isArray(result.cure) 
            ? result.cure.map(cure => `<li>${cure}</li>`).join('')
            : `<li>${result.cure}</li>`;
        
        this.cureSteps.innerHTML = `<ul>${cureList}</ul>`;
        
        this.resultsSection.style.display = 'block';
        this.resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    showLoading() {
        this.loadingSection.style.display = 'block';
        this.resultsSection.style.display = 'none';
    }

    hideLoading() {
        this.loadingSection.style.display = 'none';
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new PlantHealthDetector();
});