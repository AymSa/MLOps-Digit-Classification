import os
from flask import Flask, render_template
from flaskr import db, auth, process
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

@flow
def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY="dev",
        DATABASE=os.path.join(app.instance_path, "flaskr.sqlite"),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile("flask_config.py", silent=True)
    else:
        # load the test config if passed in
        app.config.from_pyfile(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    db.init_app(app)
    app.register_blueprint(auth.bp)
    app.register_blueprint(process.bp)
    app.add_url_rule("/", endpoint="index")

    return app

app = create_app()

if __name__ == '__main__':
    debug=True
    host='0.0.0.0'
    port=9696
    app.run(debug=debug, host=host, port=port)
