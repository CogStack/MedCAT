# A very simple proof of concept API

It requires flask to be used, set it up by running:
```
export FLASK_ENV=development
export FLASK_APP=api.py

flask run --host <ip/host>
```

NOTE: Don't forget that cat has to be in your PYTHONPATH and that you need to edit and set 
the umls path in `api/api.py`


Once it is running you can access the test page on:
`<ip>/api_test`

OR the api via:
```
curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"text": "lung cancer diagnosis"}' \
  <ip/host>
```
