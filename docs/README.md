# Docker for API

You can build and run the docker using the following process:

Cloning
```console
git clone https://github.com/jqueguiner/SRN-Deblur.git SRN-Deblur
```

Building Docker
```console
cd SRN-Deblur && docker build -t srn-deblur -f Dockerfile .
```

Running Docker
```console
echo "http://$(curl ifconfig.io):5000" && docker run -p 5000:5000 -d srn-deblur
```

Calling the API
```console
curl -X POST "http://MY_SUPER_API_IP:5000/process" -H "accept: image/*" -H "Content-Type: application/json" -d '{"url":"https://i.ibb.co/54t8zB5/input.png"}' --output deblurred.png
```