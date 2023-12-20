# hydrosmart-cc
# Description
Hydrosmart an application that that helps people cultivate hydroponic plants. This application is developed by Bangkit 2023 Capstone Project Team CH2-PS013.

# Documentation  

Cloud Services we used this following services as our infrastructure :
Cloud Run to run the back-end of the apps.  
Cloud storage to store dataset in bucket.

# Deployment:

1. First, we create a a bucket in the cloud storage using an our project service to store dataset.
2. Before deploying API on cloud run, we use Fast Api framework for a function of the Hydrosmart application, beginning with add the Method, Add the public dataset url.
3. Then, in the following step, we create a dockerfile in which this docker is used and build an image with a set of commands in Visual Studio Code.
4. Then push Image to Docker Hub Repository, along with images pushed to DockerHub. Final step is to create a cloud run service and select the images we pushed then configure the environment of Cloud Run Service.
5. Public API endpoints are now available for use in our application.

# API Documentation
Postman Documentation:https://documenter.getpostman.com/view/30300759/2s9YeLZVYm
