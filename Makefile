.PHONY: all setup run test notebook dashboard zip clean

PYTHON ?= python3

all: run

setup:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) run_pipeline.py

test:
	pytest -v

dashboard:
	streamlit run app.py

zip:
	@echo "Creating clean project ZIP archive..."
	zip -r campus_energy_forecast_project.zip . \
		-x "*.git*" \
		-x "*__pycache__*" \
		-x "*.pytest_cache*" \
		-x "*.DS_Store*" \
		-x "*.pyc"
	@echo "Archive campus_energy_forecast_project.zip created."

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache
	rm -f campus_energy_forecast_project.zip
