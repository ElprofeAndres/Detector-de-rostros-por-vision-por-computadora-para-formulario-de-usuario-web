# Importamos librerias
from flask import Flask, render_template, Response, request, redirect, url_for, session, flash
import cv2
import mediapipe as mp
from functools import wraps

# Creamos nuestra funcion de dibujo
mp_dibujo = mp.solutions.drawing_utils
conf_dibu = mp_dibujo.DrawingSpec(thickness=1, circle_radius=1)

# Creamos un objeto donde almacenaremos la malla facial
mp_malla_facial = mp.solutions.face_mesh
malla_facial = mp_malla_facial.FaceMesh(max_num_faces=1)

# Realizamos la Videocaptura
cap = cv2.VideoCapture(0)

# Creamos la app
app = Flask(__name__)
app.secret_key = 'tu_clave_secreta_aqui'  # Necesario para manejar sesiones

# Simulación de base de datos de usuarios (en producción usar una base de datos real)
users = {}

# Decorador para verificar si el usuario está logueado
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Mostramos el video en RT
def gen_frame():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = malla_facial.process(frameRGB)
        if resultados.multi_face_landmarks:
            for rostros in resultados.multi_face_landmarks:
                mp_dibujo.draw_landmarks(frame, rostros, mp_malla_facial.FACEMESH_TESSELATION, conf_dibu, conf_dibu)
        suc, encode = cv2.imencode('.jpg', frame)
        frame = encode.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Ruta principal redirige al login
@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('facial_recognition'))
    return redirect(url_for('login'))

# Ruta de login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email in users and users[email]['password'] == password:
            session['user'] = email
            return redirect(url_for('facial_recognition'))
        return render_template('login.html', error='Credenciales inválidas')
    return render_template('login.html')

# Ruta de registro
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        name = request.form['name']
        password = request.form['password']
        
        # Validaciones básicas
        if not email or not name or not password:
            return render_template('login.html', error='Todos los campos son requeridos')
            
        if email in users:
            return render_template('login.html', error='El usuario ya existe')
            
        # Registrar nuevo usuario
        users[email] = {
            'name': name,
            'password': password
        }
        
        # Iniciar sesión automáticamente
        session['user'] = email
        return redirect(url_for('facial_recognition'))
        
    return render_template('login.html')

# Ruta de reconocimiento facial
@app.route('/facial_recognition')
@login_required
def facial_recognition():
    return render_template('facial.html')

# Ruta del video
@app.route('/video')
@login_required
def video():
    return Response(gen_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Ruta para cerrar sesión
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# Ejecutamos la app
if __name__ == "__main__":
    app.run(debug=True)