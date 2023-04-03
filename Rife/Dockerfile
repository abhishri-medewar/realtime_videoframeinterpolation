
FROM nvcr.io/nvidia/pytorch:21.02-py3
ENV http_proxy 'http://proxy-chain.intel.com:911'
ENV https_proxy 'http://proxy-chain.intel.com:912'
#ENV HTTP_PROXY='http://child-prc.intel.com:913/'
#ENV HTTPS_PROXY='https://child-prc.intel.com:913/'
#ENV FTP_PROXY='http://child-prc.intel.com:913/'
#ENV NO_PROXY='intel.com,.intel.com,10.0.0.0/8,192.168.0.0/16,localhost,.local,127.0.0.0/8,134.134.0.0/16'

RUN pip install numpy>=1.16
RUN pip install numpy>=1.16
RUN pip install tqdm>=4.35.0
RUN pip install sk-video>=1.1.10
RUN pip install opencv-python>=4.1.2
RUN pip install moviepy>=1.0.3

