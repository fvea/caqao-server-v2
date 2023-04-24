from flask import Flask, request, send_file, jsonify, make_response
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from PIL import Image
import socket
import datetime
import jwt

from urllib.parse import urlparse
import torch
torch.set_num_threads(1)
import io
import os
import secrets
import uuid
import random
import tensorflow as tf

from db import db_init, db
from model import Detection, TempDetection, User


app = Flask(__name__)
# SQLAlchemy config. Read more: https://flask-sqlalchemy.palletsprojects.com/en/2.x/
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///detections.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db_init(app)

app.config.update(SESSION_COOKIE_SAMESITE="None", SESSION_COOKIE_SECURE=True)

MAX_DET = 50
model = torch.hub.load('ultralytics/yolov5', 'custom', path="best.pt", force_reload=True)
model.max_det = MAX_DET

cacao_image_classifier = tf.keras.models.load_model('xception_v2.h5')

@app.route("/assess", methods=["POST"])
def assess():

    if request.method == "POST":

        # get cacao beans image and bean size from request
        image_file = request.files.get("image")
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        bean_size = int(request.form["beanSize"])

        # check if image is a cacao beans in a guillotine
        if not is_cacao(image):
            # temporarily save the detection result
            temp_detection = TempDetection(
                image=image_bytes,
                mimetype=image_file.mimetype,
                filename=str(uuid.uuid4()) + str(random.randint(1, 10000)) + '.jpg',
                beanGrade="--",
            )
            db.session.add(temp_detection)
            db.session.commit()
            return get_json_response(temp_detection)

        # reduce size=640 for faster inference
        results = model(image, size=640)
        results.render()
        image_detection = Image.fromarray(results.ims[0])

        # convert the image detection results to bytes
        with io.BytesIO() as output_bytes:
            image_detection.save(output_bytes, format='JPEG')
            image_detection_bytes = output_bytes.getvalue()

        # get cacao class counts and compute bean grade
        class_counts = get_class_detection_counts(results)
        bean_grade = get_bean_grade(class_counts, bean_size)

        # temporarily save the detection result
        temp_detection = TempDetection(
            image=image_detection_bytes,
            mimetype=image_file.mimetype,
            filename=str(uuid.uuid4()) + str(random.randint(1, 10000)) + '.jpg',
            beanGrade=bean_grade,
            **class_counts
        )
        db.session.add(temp_detection)
        db.session.commit()

    return get_json_response(temp_detection)

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']
            print(token)
        if not token:
            return jsonify({'message' : 'Token is missing!'}), 401
        try: 
            current_user = User.query.filter_by(public_id=request.headers['x-access-token']).first()
            print(current_user.id)
        except:
            return jsonify({'message' : 'Token is invalid!'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

@app.route("/save_results", methods=["POST"])
@token_required
def save_results(current_user):

    if request.method == "POST":
        # save the temporary detection to main detection table

        image_src_url = str(request.form["imgSrcUrl"])
        filename = os.path.basename(urlparse(image_src_url).path)

        user_id = current_user.id

        temp_detection = TempDetection.query.filter_by(filename=filename).first()
        detection = Detection(
            user_id=user_id,
            image=temp_detection.image,
            mimetype=temp_detection.mimetype,
            filename=temp_detection.filename,
            beanGrade=temp_detection.beanGrade,
            veryDarkBrown = temp_detection.veryDarkBrown,
            brown = temp_detection.brown,
            partlyPurple = temp_detection.partlyPurple,
            totalPurple = temp_detection.totalPurple,
            g1 = temp_detection.g1,
            g2 = temp_detection.g2,
            g3 = temp_detection.g3,
            g4 = temp_detection.g4,
            mouldy = temp_detection.mouldy,
            insectDamaged = temp_detection.insectDamaged,
            slaty = temp_detection.slaty,
            germinated = temp_detection.germinated,
            date = temp_detection.date
        )
        # add the new record to the database session and commit the changes
        db.session.add(detection)
        db.session.commit()
        # delete the temporary detections
        db.session.delete(temp_detection)
        db.session.commit()

        return "Assessment Results Saved", 200

@app.route('/get_detection_with_id', methods=["POST"])
@token_required
def get_detection_with_id(current_user):
    if request.method == "POST":
        id = int(request.form["cacaoDetectionId"])
        detection = Detection.query.filter_by(user_id=current_user.id, id=id).first()
        return get_json_response(detection)


@app.route('/detections')
@token_required
def get_detections(current_user):
    user_id = current_user.id
    detections = Detection.query.filter_by(user_id=user_id).order_by(Detection.date.desc()).all()
    detection_list = [
        {'id': detection.id, \
         'img_src_url': f"http://{FLASK_IP_ADDR}:5000/detections/{detection.filename}", \
         'g1': detection.g1, \
         'g2': detection.g2, \
         'g3': detection.g3, \
         'g4': detection.g4, \
         'veryDarkBrown': detection.veryDarkBrown, \
         'brown': detection.brown, \
         'partlyPurple': detection.partlyPurple, \
         'totalPurple': detection.totalPurple, \
         'mouldy': detection.mouldy, \
         'insectInfested': detection.insectDamaged, \
         'slaty': detection.slaty,\
         'germinated': detection.germinated, \
         'beanGrade': detection.beanGrade, \
         'date': detection.date } for detection in detections]
    return jsonify(detection_list)

@app.route('/detections/<string:filename>')
def get_image(filename):
    detection = Detection.query.filter_by(filename=filename).first()
    if not detection:
        detection = TempDetection.query.filter_by(filename=filename).first()

    return send_file(io.BytesIO(detection.image), mimetype=detection.mimetype, download_name=detection.filename)

@app.route('/validate_image', methods=['POST'])
def validate_image():
    
    # get cacao beans image and bean size from request
    image_file = request.files.get("image")
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # check if image is a cacao beans in a guillotine
    if not is_cacao(image):
        response = {
            "message": "Invalid Image: Image must contain cacao beans in a guillotine.",
            "status": 400
        }
        return jsonify(response)

    else:
        response = {
            "message": "Valid Image",
            "status": 200
        }
        return jsonify(response)
    
            
@app.route('/register', methods=['POST'])
def create_user():
    data = request.get_json()
    hashed_password = generate_password_hash(data['password'], method='sha256')

    if is_username_exists(data['username']):
        # username is already taken
       return jsonify({'message' : 'Username is already taken.', 'status': 401})
    
    if is_user_email_exists(data['email']):
        # email is already taken
        return jsonify({'message' : 'Email is already taken.', 'status': 402})
    
    # create and add new user to database
    new_user = User(
        public_id=str(uuid.uuid4()), 
        first_name=data['first_name'], 
        last_name=data['last_name'],
        email=data['email'],
        username=data['username'],
        password=hashed_password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message' : 'New user created!', 'status': 200})
    
@app.route('/login', methods=['POST'])
def login():

    # get authentication data
    auth = request.get_json()

    # get user data from database
    user = User.query.filter_by(username=auth['username']).first()

    if not user:
        return jsonify({'message' : 'Wrong email or password', 'status': 401, 'token': ''})

    # check if entered hashed password matches the hashed password in the database
    if check_password_hash(user.password, auth['password']):
        token = user.public_id
        return jsonify({'message' : 'User Authenticated', 'status': 200, 'token': token})

    return jsonify({'message' : 'Wrong email or password', 'status': 401, 'token': ''})

@app.route('/delete', methods=['POST'])
@token_required
def delete_detection(current_user):
    
    record_id = int(request.form["cacaoDetectionId"])
    user_id = current_user.id

    detection = Detection.query.filter_by(id=record_id, user_id=user_id).first()

    if detection is None:
        return 'Record not found', 404
    
    db.session.delete(detection)
    db.session.commit()
    return 'Record deleted', 200

@app.route('/')
def index():
    return "CAQAO Server"

def get_json_response(detection):
    json_response = {
        'id': detection.id,
        'img_src_url': f"http://{FLASK_IP_ADDR}:5000/detections/{detection.filename}",
        "veryDarkBrown" : detection.veryDarkBrown,
        "brown" : detection.brown,
        "partlyPurple" : detection.partlyPurple,
        "totalPurple" : detection.totalPurple,
        "g1" : detection.g1,
        "g2" : detection.g2,
        "g3" : detection.g3,
        "g4" : detection.g4,
        "mouldy" : detection.mouldy,
        "insectInfested" : detection.insectDamaged,
        "slaty" : detection.slaty,
        "germinated" : detection.germinated,
        "beanGrade": detection.beanGrade,
        "date": detection.date
    }
    return jsonify(json_response)

def get_class_detection_counts(results):
    class_counts = {
        "veryDarkBrown" : 0,
        "brown" : 0,
        "partlyPurple" : 0,
        "totalPurple" : 0,
        "g1" : 0,
        "g2" : 0,
        "g3" : 0,
        "g4" : 0,
        "mouldy" : 0,
        "insectDamaged" : 0,
        "slaty" : 0,
        "germinated" : 0,
    }

    for key, value in results.pandas().xyxy[0].name.value_counts().items():
        key = key.lower()
        key_split = key.split("-")
        if len(key_split) > 1:
            
            # handle insect-damaged
            if key_split[0] == 'insect':
                class_counts['insectDamaged'] += value
                continue

            color, grade = key_split[0], key_split[1]
            color = color.split()
            if len(color) == 0:
                class_counts[color.strip()] += value
                class_counts[grade.strip()] += value
            else:
                color = color[0].lower() + ''.join(i.capitalize() for i in color[1:])
                class_counts[color.strip()] += value
                class_counts[grade.strip()] += value
        else:
            defect = key_split[0]
            defect = defect.split()
            if len(defect) == 0:
                class_counts[defect.strip()] += value
            else:
                defect = defect[0].lower() + ''.join(i.capitalize() for i in defect[1:])
                class_counts[defect.strip()] += value

    return class_counts

def get_bean_grade(class_count, bean_size):
    slaty, mouldy = class_count["slaty"], class_count["mouldy"]
    insect_infested, germinated = class_count["insectDamaged"], class_count["germinated"]
    letter_code, num_code = "", ""
    tresholdNumCode = 0.03 * MAX_DET

    if (slaty <= tresholdNumCode) and (mouldy <= tresholdNumCode) and \
        ((insect_infested + germinated) <= tresholdNumCode):
            num_code = "1"
    else:
            num_code = "2"

    if bean_size <= 100:
        letter_code = "A"
    elif (bean_size >= 101) and (bean_size <= 110):
        letter_code = "B"
    else:
        letter_code = "C"

    return num_code + letter_code

def is_cacao(img, threshold=0.95):

    class_names = ['cacao', 'noncacao']
    img_resized = img.resize((150, 150))

    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0) # create a bactch

    # img_array = img_array.reshape(None, 150, 150, 3)

    raw_prediction = cacao_image_classifier.predict(img_array)[0][0]
    prediction = 1 if raw_prediction > threshold else 0
    class_name = class_names[prediction]
    confidence = (1 - raw_prediction) * 100 if prediction == 0 else (raw_prediction * 100)

    return class_name == 'cacao'

def is_username_exists(username):
    user_via_username = User.query.filter_by(username=username).first()
    if user_via_username:
        return True
    else:
        return False
    
def is_user_email_exists(email):
    user_via_email = User.query.filter_by(email=email).first()
    if user_via_email:
        return True
    else:
        return False
    
if __name__ == "__main__":

    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    FLASK_IP_ADDR = ip_address

    MAX_DET = 50
    model.max_det = MAX_DET
    app.run(host="0.0.0.0", port=5000, debug=True)