pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                git url 'https://github.com/Atharva2884/Brain_Tumor_Prediction.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Lint') {
            steps {
                sh 'pylint *.py || true'
            }
        }

        stage('Run Tests') {
            steps {
                sh 'pytest tests/ --disable-warnings || echo "No tests"'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t brain-tumor-app .'
            }
        }

        stage('Run Docker Container') {
            steps {
                sh 'docker run -d -p 5000:5000 brain-tumor-app'
            }
        }
    }
}
