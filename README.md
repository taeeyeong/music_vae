
# Music VAE Replication

Music VAE Replication 
paper: https://arxiv.org/pdf/1803.05428.pdf

## Getting Started

This section should provide instructions on how to get a copy of the project up and running on a local machine for development and testing purposes.

### Prerequisites

List all the prerequisites, the things you need to install the software, and how to install them.

```
Give examples
```

### Installing

프로젝트를 실행하기 전에 필요한 환경 설정을 설치합니다. 

```
pip install -r requirements.txt
```

And repeat:

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo.

## Preprocessing

MIDI 파일을 사용하여 음악을 생성하기 위해 데이터를 전처리해야 합니다. 

MIDI 파일 경로와 전처리된 데이터를 저장할 파일 경로를 지정해야 합니다.

```
python preprocess.py --input_file input.mid --output_file output.pkl --seq_length 100
```
여기서,
- input.mid: 전처리할 MIDI 파일의 경로입니다.
- output.pkl: 전처리된 데이터를 저장할 파일의 경로입니다.
- seq_length: 전처리할 시퀀스 길이입니다. 기본값은 100입니다.

전처리된 데이터는 pickle 형식으로 저장됩니다.


## Running the Tests

Explain how to run the automated tests for this system.

### Break down into end-to-end tests

Explain what these tests test and why.

```

```

### And coding style tests

Explain what these tests test and why.

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system.

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://github.com/your/project/contributing.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).

## Authors

* **Your Name** - *Initial work* - [YourUsername](https://github.com/YourUsername)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the XYZ License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
