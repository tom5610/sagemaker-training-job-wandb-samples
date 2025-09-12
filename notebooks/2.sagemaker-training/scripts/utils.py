import boto3
import json
import time
from botocore.exceptions import ClientError

# Initialize the Secrets Manager client
secretsmanager = boto3.client('secretsmanager')

def create_role(role_name):
    """
    Create a SageMaker execution role with the necessary permissions
    
    Args:
        role_name (str): Name of the IAM role to create
    
    Returns:
        str: ARN of the created role
    """
    iam_client = boto3.client('iam')
    
    # Define trust relationship policy for SageMaker
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "sagemaker.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    try:
        # Check if the role already exists
        try:
            response = iam_client.get_role(RoleName=role_name)
            print(f"Role {role_name} already exists.")
            return response['Role']['Arn']
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchEntity':
                # Role doesn't exist, create it
                response = iam_client.create_role(
                    RoleName=role_name,
                    AssumeRolePolicyDocument=json.dumps(trust_policy),
                    Description='SageMaker execution role for training jobs with Weights & Biases integration'
                )
                print(f"Created role: {role_name}")
                
                # Wait for the role to be created
                print("Waiting for the role to be available...")
                time.sleep(10)
                
                role_arn = response['Role']['Arn']
                return role_arn
            else:
                raise
    except Exception as e:
        print(f"Error creating role: {str(e)}")
        raise

def create_custom_policy(role_name):
    """
    Create and attach custom policy for tagging SageMaker training jobs
    
    Args:
        role_name (str): Name of the IAM role
    """
    iam_client = boto3.client('iam')
    
    # Custom policy for tagging SageMaker training jobs
    tagging_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "VisualEditor0",
                "Effect": "Allow",
                "Action": [
                    "sagemaker:AddTags",
                    "sagemaker:ListTags"
                ],
                "Resource": "*"
            },
            {
                "Sid": "AllowSecretManagerActions",
                "Effect": "Allow",
                "Action": [
                    "secretsmanager:DescribeSecret",
                    "secretsmanager:GetSecretValue",
                    "secretsmanager:CreateSecret"
                ],
                "Resource": [
                    "arn:aws:secretsmanager:*:*:secret:wandb-*"
                ]
            },
            {
                "Sid": "AllowS3ObjectActions",
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:AbortMultipartUpload"
                ],
                "Resource": [
                    "arn:aws:s3:::*SageMaker*",
                    "arn:aws:s3:::*Sagemaker*",
                    "arn:aws:s3:::*sagemaker*"
                ]
            },
            {
                "Sid": "AllowAWSServiceActions",
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogDelivery",
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:DeleteLogDelivery",
                    "logs:Describe*",
                    "logs:GetLogDelivery",
                    "logs:GetLogEvents",
                    "logs:ListLogDeliveries",
                    "logs:PutLogEvents",
                    "logs:PutResourcePolicy",
                    "logs:UpdateLogDelivery",
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:BatchGetImage",
                    "ecr:CreateRepository",
                    "ecr:Describe*",
                    "ecr:GetAuthorizationToken",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:StartImageScan",
                    "cloudwatch:DeleteAlarms",
                    "cloudwatch:DescribeAlarms",
                    "cloudwatch:GetMetricData",
                    "cloudwatch:GetMetricStatistics",
                    "cloudwatch:ListMetrics",
                    "cloudwatch:PutMetricAlarm",
                    "cloudwatch:PutMetricData"
                ],
                "Resource": "*"
            }
        ]
    }
    
    policy_name = f"misc-policy"
    
    try:
        # Check if the policy already exists
        policies = iam_client.list_policies(Scope='Local')
        policy_exists = False
        policy_arn = None
        
        for policy in policies['Policies']:
            if policy['PolicyName'] == policy_name:
                policy_exists = True
                policy_arn = policy['Arn']
                break
        
        if not policy_exists:
            # Create the policy
            response = iam_client.create_policy(
                PolicyName=policy_name,
                PolicyDocument=json.dumps(tagging_policy),
                Description='Custom policy for SageMaker training jobs execution.'
            )
            policy_arn = response['Policy']['Arn']
            print(f"Created custom policy: {policy_name}")
        else:
            print(f"Custom policy {policy_name} already exists")
        
        # Attach the policy to the role
        try:
            iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn=policy_arn
            )
            print(f"Attached custom policy {policy_name} to role {role_name}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'EntityAlreadyExists':
                print(f"Policy {policy_name} is already attached to role {role_name}")
            else:
                raise
    except Exception as e:
        print(f"Error with custom policy: {str(e)}")
        raise

def create_sagemaker_execution_role(role_name:str) -> str:
    
    print(f"Creating SageMaker execution role: {role_name}")
    role_arn = create_role(role_name)
    
    print(f"Creating and attaching custom misc policy to role: {role_name}")
    create_custom_policy(role_name)
    
    print(f"\nSageMaker execution role created successfully!")
    print(f"Role ARN: {role_arn}")

    return role_arn


def create_wandb_secret(wandb_secret_name:str, wandb_api_key: str):

    # Create the secret with WANDB_API_KEY
    secret_value = {"WANDB_API_KEY": wandb_api_key}
    secret_string = json.dumps(secret_value)

    # Create the secret in AWS Secrets Manager
    try:
        # Try to get the secret to check if it exists
        secretsmanager.get_secret_value(SecretId=wandb_secret_name)
        
        # Secret exists, update it
        response = secretsmanager.update_secret(
            SecretId=wandb_secret_name,
            SecretString=secret_string
        )
        print(f"Secret updated successfully: {wandb_secret_name}")
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            # Secret doesn't exist, create it
            try:
                response = secretsmanager.create_secret(
                    Name=wandb_secret_name,
                    SecretString=secret_string
                )
                print(f"Secret created successfully: {response['ARN']}")
            except Exception as create_error:
                print(f"Error creating secret: {str(create_error)}")
        else:
            # Other error occurred
            print(f"Error accessing secret: {str(e)}")    


def create_s3_bucket(bucket_name: str, region=None):
    """
    Create an S3 bucket if it doesn't already exist
    
    Args:
        bucket_name (str): Name of the S3 bucket to create
        region (str, optional): AWS region to create the bucket in. Defaults to None (uses boto3 default region).
    
    Returns:
        str: Name of the bucket
    """
    s3_client = boto3.client('s3', region_name=region)
    
    try:
        # Check if the bucket already exists
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"S3 bucket {bucket_name} already exists.")
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            # Bucket doesn't exist, create it
            try:
                if region is None:
                    s3_client.create_bucket(Bucket=bucket_name)
                else:
                    location = {'LocationConstraint': region}
                    s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration=location
                    )
                print(f"Created S3 bucket: {bucket_name}")
            except ClientError as create_error:
                print(f"Error creating bucket: {str(create_error)}")
                raise
        elif error_code == '403':
            print(f"Access forbidden: You don't have permission to access the bucket {bucket_name}")
            raise
        else:
            print(f"Error checking bucket: {str(e)}")
            raise
            
    return bucket_name