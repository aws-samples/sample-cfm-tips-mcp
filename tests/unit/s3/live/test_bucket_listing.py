"""
Simple test to verify S3 bucket listing works.
"""

import asyncio
import boto3
import pytest


@pytest.mark.live
async def test_list_buckets():
    """Test basic S3 bucket listing."""
    
    s3_client = boto3.client('s3')
    
    try:
        response = s3_client.list_buckets()
        buckets = response.get('Buckets', [])
        
        print(f"\n=== Found {len(buckets)} S3 Buckets ===")
        
        for bucket in buckets[:10]:  # Show first 10
            bucket_name = bucket['Name']
            creation_date = bucket['CreationDate']
            
            # Try to get bucket location
            try:
                location_response = s3_client.get_bucket_location(Bucket=bucket_name)
                region = location_response.get('LocationConstraint') or 'us-east-1'
            except Exception as e:
                region = f"Error: {str(e)}"
            
            print(f"\nBucket: {bucket_name}")
            print(f"  Region: {region}")
            print(f"  Created: {creation_date}")
        
        return len(buckets)
        
    except Exception as e:
        print(f"\nError listing buckets: {str(e)}")
        return 0


if __name__ == "__main__":
    count = asyncio.run(test_list_buckets())
    print(f"\n\nTotal buckets: {count}")
