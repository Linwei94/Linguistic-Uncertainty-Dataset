import boto3
import json


class qualification:
    def __init__(self):
        self.mturk = None

    def connectProcess(self, login_info):
        self.connectAws(login_info)

    def registerProcess(self, qualificationTypeParams):
        self.qualificationTypeId = self.createQualificationType(qualificationTypeParams)

    def connectAws(self, login_info):
        self.mturk = boto3.client(
            'mturk',
            aws_access_key_id=login_info['aws_access_key_id'],
            aws_secret_access_key=login_info['aws_secret_access_key'],
            region_name=login_info['region_name'],
            endpoint_url=login_info['endpoint_url']
        )
        print("Account balance:", self.mturk.get_account_balance()['AvailableBalance'])
        return self.mturk

    def createQualificationType(self, qualificationTypeParams):
        qualification_type_id = {}
        for param in qualificationTypeParams:
            try:
                response = self.mturk.create_qualification_type(
                    Name=param['Name'],
                    Keywords=param['Keywords'],
                    Description=param['Description'],
                    QualificationTypeStatus=param['QualificationTypeStatus'],
                )
                qualification_type_id[param['Name']] = response['QualificationType']['QualificationTypeId']
                print(f"Successly create qualification type {param['Name']}: {qualification_type_id[param['Name']]}")
            except Exception as e:
                print(f"Error create qualification type {param['Name']}: {e}")
                qualification_type_id[param['Name']] = None

        with open("qualification.json", "w") as f:
            json.dump(qualification_type_id, f, indent=2)
        return qualification_type_id


def testQualification():
    qua = qualification()
    # # 连接aws
    login_info = {
        'aws_access_key_id': '',
        'aws_secret_access_key': '',
        'region_name': 'us-east-1',
        'endpoint_url': 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
    }
    qua.connectProcess(login_info)

    # 注册qualification
    qualificationTypeParams = [
        {
            'Name': "Max5Assignments2",
            'Keywords': "max5, control, limit",
            'Description': "Limits a worker to at most 5 assignments.",
            'QualificationTypeStatus': "Active",
        }
    ]
    qua.registerProcess(qualificationTypeParams)


if __name__ == '__main__':
    testQualification()
