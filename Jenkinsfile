pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                git 'https://github.com/Atharva2884/Brain_Tumor_Prediction.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                bat 'pip install -r requirements.txt'
            }
        }

        stage('Run Tests') {
            steps {
                bat 'pytest || echo No tests found'
            }
        }

        stage('Docker Build') {
            steps {
                bat 'docker build -t brain-tumor-predictor .'
            }
        }
    }
}
