all: clean build test install

clean:
	rm -Rf *.egg-info
	rm -Rf build
	rm -Rf dist
	rm -Rf .pytest_cache
	rm -f .coverage

build: clean
	python3 setup.py sdist

#tests:
#	flake8 -vv bazema_pokemon
#	coverage run --source=bazema_pokemon -m pytest
#	coverage report --omit="*/test*"

linter:
	pip install -r requirements.txt
	pip install pylint
	pylint bazema_pokemon

install: clean
	pip install torch torchvision torchaudio opencv-contrib-python
	python setup.py install

deploy-pip: install
	pip install twine
	twine upload -u bameza -p ${PYPI_PWD} --skip-existing dist/*

pdf:
	docker run --rm -v $PWD:/app jmaupetit/md2pdf --css pdf/style.css pdf/project_report.md pdf/project_report.pdf
	docker run --rm -v $PWD:/app jmaupetit/md2pdf --css pdf/style.css pdf/proposal.md pdf/proposal.pdf