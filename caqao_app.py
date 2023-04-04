from flask import Flask, request, send_file, jsonify
from werkzeug.security import check_password_hash, generate_password_hash
from PIL import Image
import socket
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity, create_access_token


from urllib.parse import urlparse
import torch
torch.set_num_threads(1)
import io
import os
import secrets
import uuid
import random


from db import db_init, db
from model import Detection, TempDetection, User


app = Flask(__name__)
# SQLAlchemy config. Read more: https://flask-sqlalchemy.palletsprojects.com/en/2.x/
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///detections.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db_init(app)

jwt = JWTManager(app)

app.config.update(SESSION_COOKIE_SAMESITE="None", SESSION_COOKIE_SECURE=True)

MAX_DET = 50
model = torch.hub.load('ultralytics/yolov5', 'custom', path="best.pt", force_reload=True)
model.max_det = MAX_DET

@app.route("/assess", methods=["POST"])
def assess():

    if request.method == "POST":

        # get cacao beans image and bean size from request
        image_file = request.files.get("image")
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        bean_size = int(request.form["beanSize"])

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


@app.route("/save_results", methods=["POST"])
@jwt_required()
def save_results():

    if request.method == "POST":
        # save the temporary detection to main detection table

        image_src_url = str(request.form["imgSrcUrl"])
        filename = os.path.basename(urlparse(image_src_url).path)

        user_id = get_jwt_identity()

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


@app.route('/detections')
def get_detections():

    user_id = 1
    detections = Detection.query.filter_by(user_id=user_id).all()

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

@app.route('/recent_detections')
@jwt_required()
def get_recent_detections():

    user_id = get_jwt_identity()

    query = Detection.query.filter_by(user_id=user_id).order_by(Detection.date.desc())

    # get the first five recent detections
    records = query[:5]

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
         'date': detection.date } for detection in records]
    
    return jsonify(detection_list)


# define endpoint for creating a new user account
@app.route('/create_user', methods=['POST'])
def create_user():
    
    # get user information from request body
    first_name = str(request.form["firstName"])
    last_name = str(request.form["lastName"])
    username = str(request.form["username"])
    email = str(request.form["email"])
    password = generate_password_hash(str(request.form["password"]))    

    # save user data to database
    new_user = User(
        first_name=first_name,
        last_name=last_name,
        username=username,
        email=email,
        password=password
    )

    # add the new record to the database session and commit the changes
    db.session.add(new_user)
    db.session.commit()

    # return success message
    return jsonify({'message': 'User account created successfully'}), 201


@app.route('/login', methods=['POST'])
def login():

    username = str(request.form.get('username'))
    password = str(request.form.get('password'))

    # Verify user credentials
    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password, password):
        # Generate access token
        access_token = create_access_token(identity=user.id)
        return jsonify({'access_token': access_token, 'status': 200})
    else:
        return jsonify({'access_token': '--', 'status': 401})


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


if __name__ == "__main__":

    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    FLASK_IP_ADDR = ip_address

    MAX_DET = 50
    model.max_det = MAX_DET
    app.run(host="0.0.0.0", port=5000, debug=True)