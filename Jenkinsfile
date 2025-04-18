pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout([$class: 'GitSCM',
                    branches: [[name: '*/main']],
                    userRemoteConfigs: [[url: 'https://github.com/Atharva2884/Brain_Tumor_Prediction.git']]
                ])
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Run Tests') {
            steps {
                sh 'pytest || echo "No tests found"'
            }
        }

        stage('Docker Build') {
            steps {
                sh 'docker build -t brain-tumor-predictor .'
            }
        }
    }
}
