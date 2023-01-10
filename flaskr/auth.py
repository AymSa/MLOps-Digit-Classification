from functools import wraps

from flask import (
    Blueprint,
    flash,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import (
    check_password_hash,
    generate_password_hash,
)  # authentification securisé

from flaskr.db import get_db

bp = Blueprint("auth", __name__, url_prefix="/auth")


@bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":  # C'est a dire que l'user s'est inscrit
        username = request.form["username"]
        password = request.form["password"]
        # On recuperer les identifiants + mdp
        db = (
            get_db()
        )  # on recupere la base de données (la connexion avec la db reste active pendant la requete)

        error = None

        if not username:
            error = "Username is required."
        elif not password:
            error = "Password is required."

        if error is None:
            try:
                db.execute(  # Lance une commande SQL pour ajouter les users et passwords
                    "INSERT INTO user (username, password) VALUES (?, ?)",
                    (username, generate_password_hash(password)),
                )
                db.commit()
            except db.IntegrityError:
                error = f"User {username} is already registered."
            else:
                return redirect(url_for("auth.login"))

        flash(error)

    return render_template("auth/register.html")


@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":  # C'est a dire que l'user a mis ses identifiants + mdp
        username = request.form["username"]
        password = request.form["password"]
        # On créer un dict qui va recuperer les identifiants + mdp
        db = (
            get_db()
        )  # on recupere la base de données (la connexion avec la db reste active pendant la requete)
        user = db.execute(
            "SELECT * FROM user WHERE username = ?", (username,)
        ).fetchone()
        error = None

        if not username:
            error = "Username is required."
        elif not password:
            error = "Password is required."

        if user is None:
            error = "Incorrect username."
        elif not check_password_hash(user["password"], password):
            error = "Incorrect password."

        if error is None:
            session.clear()
            session["user_id"] = user["id"]  # unique id generer incrementalement
            return redirect(url_for("index"))

        flash(error)

    return render_template("auth/login.html")


@bp.before_app_request  # Avant chaque requete on stocke l'id de l'user dans le contexte ???? C'est pas auto ?
def load_logged_in_user():
    user_id = session.get("user_id")

    if user_id is None:
        g.user = None
    else:
        g.user = (
            get_db().execute("SELECT * FROM user WHERE id = ?", (user_id,)).fetchone()
        )


@bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


# Creation d'un decorateur pour verifier qu'on est bien connecter avant chaque action


def login_required(view):
    @wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return redirect(url_for("auth.login"))

        return view(**kwargs)

    return wrapped_view
