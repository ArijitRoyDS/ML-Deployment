# ML-Deployment
This repository contains Machine Learning Deployment projects

General Instructions:

Follow the below steps to deploy your first Machine Learning Model using Amazon Web Services which is a popular IaaS (Infrastructure As A Service).

Prerequisites:
1) Trained Model exported as a pickle file (*.pkl)
2) Flask script to develop a basic web interface to accept inputs from users, predict the output using the pickle file and show the output
3) requirements.txt file with all the required libraries
4) AWS Account (Free Version) - https://aws.amazon.com/free
5) Install PuTTy - https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html
6) Install WinSCP - https://winscp.net/eng/download.php

Steps for deployment:

Step 1: Log In to AWS and Create a Virtual Machine
	1) Search for EC2 and click on Instances >> Launch an Instance
	2) Provide a name and choose Ubuntu Server 20.04 LTS as the OS
	3) Create a new Key Pair, Provide a name, Type = RSA, File Format = .pem and save it locally
	4) Click on Launch Instance >> Connect to Instance >> Connect
	5) Go to EC2 >> Instances (running), select the instance and click on Connect
	6) Under 'EC2 Instance Connect' tab, there is a user name 'ubuntu' note it.
	7) Under SSH Client tab, copy the Public DNS under item no 4

Step 2: Generate Private Key using PuTTY Key Generator
	- Launch PuTTY Key Generator on your local machine
	- Click on Load and browse for the .pem file generated in Step 1
	- Click on Save Private Key and save the Putty Key (*.ppk) locally

Step 3: Mount the Virtual Machine on WinSCP Client using the Private Key:
	- Launch WinSCP on local machine
	- Click on New Site and fill up the Host Name with the Public DNS & user name copied in Step 1
	- Click on Advanced >> SSH >> Authentication and upload the *.ppk file generated in Step 2
	- Click on OK and Login

Step 4: Connect to the Virtual Machine using PuTTY
	- Launch PuTTY on local machine and fill up the Host Name with the Public DNS copied in Step 1
	- On the left side menu, click on Connection >> SSH >> Auth >> Credentials and upload the Private Key generated in Step 2
	- Login with the default username 'ubuntu'. Password is not required.
	- Install all required Python Modules and execute the application on the virtual machine to test if everything works fine

Step 5: Configure Security Groups on AWS
	- Login to your AWS and under Network & Security, click on Security Groups
	- Click on Create Security Group. Provide a Name and Description
	- Under Inbound Rule, click on Add Rule. 
	- Under Type, select 'All Traffic' & under Source, select 'Anywhere IPv4'. Click on Create Security Group.
	- Go to Network & Security >> Network Interfaces. Right click on Security Group Names >> Change Security Group
	- Choose the security group that we just created and remove the default security group pre-assigned

All the setup is done. 

Use the Public DNS copied from Step 1, followed by port No 8080
For example: ec2-35-154-243-236.ap-south-1.compute.amazonaws.com:8080
