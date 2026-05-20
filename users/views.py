from django.shortcuts import render, HttpResponse
from django.contrib import messages
from django.conf import settings

from .forms import UserRegistrationForm
from .models import UserRegistrationModel

import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score, roc_curve

trained_model = None


# ---------------- User Registration ----------------
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'You have been successfully registered')
            return render(request, 'UserRegistrations.html', {'form': UserRegistrationForm()})
        messages.error(request, 'Email or Mobile already exists')
    return render(request, 'UserRegistrations.html', {'form': UserRegistrationForm()})


# ---------------- User Login ----------------
def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid').strip().lower()
        pswd = request.POST.get('pswd').strip()

        user = UserRegistrationModel.objects.filter(
            loginid=loginid,
            password=pswd
        ).first()

        if not user:
            messages.error(request, "Invalid Login ID or Password")
            return render(request, 'UserLogin.html')

        
        if user.status.strip().lower() not in ["activated", "accepted"]:
            messages.error(request, "Your account is not activated")
            return render(request, 'UserLogin.html')

        request.session['id'] = user.id
        request.session['loggeduser'] = user.name
        request.session['loginid'] = user.loginid
        request.session['email'] = user.email

        return render(request, 'users/UserHomePage.html')

    return render(request, 'UserLogin.html')


def UserHome(request):
    return render(request, 'users/UserHomePage.html')


# ---------------- Dataset View ----------------
def DatasetView(request):
    df = pd.read_csv(settings.MEDIA_ROOT + "/balanced_dataset.csv")
    return render(request, 'users/viewdataset.html', {'data': df.to_html()})


# ---------------- Training ----------------
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve

def Training(request):
    global trained_model

    df = pd.read_csv(settings.MEDIA_ROOT + "/alzheimers_disease_data.csv")

    FEATURES = [
        'Age', 'MMSE', 'FunctionalAssessment', 'ADL', 'EducationLevel',
        'CholesterolTotal', 'MemoryComplaints', 'Confusion',
        'Disorientation', 'Forgetfulness', 'DifficultyCompletingTasks',
        'PersonalityChanges', 'FamilyHistoryAlzheimers',
        'Depression', 'HeadInjury'
    ]

    X = df[FEATURES]
    y = df['Diagnosis']

    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_cols),

        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ])

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=8,
            min_samples_leaf=10, random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, random_state=42
        ),
        'SVM': SVC(kernel='rbf', probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=7),
        'XGBoost': XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42
        )
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    # ---------- Evaluate all models ----------
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('model', model)
        ])

        acc = cross_val_score(
            pipeline, X, y, cv=cv, scoring='accuracy'
        ).mean()

        auc = cross_val_score(
            pipeline, X, y, cv=cv, scoring='roc_auc'
        ).mean()

        results[name] = {
            'accuracy': round(acc, 4),
            'auc': round(auc, 4),
            'pipeline': pipeline
        }

    # ---------- Select best model ----------
    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    best_pipeline = results[best_model_name]['pipeline']

    # ---------- ROC CURVE (BEST MODEL ONLY, CV-BASED) ----------
    y_proba = cross_val_predict(
        best_pipeline,
        X,
        y,
        cv=cv,
        method='predict_proba'
    )[:, 1]

    best_fpr, best_tpr, _ = roc_curve(y, y_proba)
    best_auc = roc_auc_score(y, y_proba)

    # ---------- Train best model on full data ----------
    best_pipeline.fit(X, y)

    trained_model = {
        'name': best_model_name,
        'model': best_pipeline
    }

    context = {
        'results': {k: v['accuracy'] for k, v in results.items()},
        'auc_scores': {k: v['auc'] for k, v in results.items()},
        'best_model': best_model_name,
        'best_accuracy': results[best_model_name]['accuracy'],
        'roc_curve': {
            'fpr': best_fpr.tolist(),
            'tpr': best_tpr.tolist(),
            'auc': round(best_auc, 4)
        }
    }

    return render(request, 'users/training.html', context)


# ---------------- Prediction ----------------
def yn(val):
    return 1 if val == 'Yes' else 0


def Prediction(request):
    global trained_model

    if trained_model is None:
        messages.error(request, "Please train the model first")
        return render(request, 'users/predict_form.html')

    if request.method == "POST":
        input_data = {
            'Age': float(request.POST['age']),
            'MMSE': float(request.POST['mmse']),
            'FunctionalAssessment': float(request.POST['functional_assessment']),
            'ADL': float(request.POST['adl']),
            'EducationLevel': int(request.POST['education_level']),
            'CholesterolTotal': float(request.POST['CholesterolTotal']),
            'MemoryComplaints': yn(request.POST['memory_complaints']),
            'Confusion': yn(request.POST['confusion']),
            'Disorientation': yn(request.POST['disorientation']),
            'Forgetfulness': yn(request.POST['forgetfulness']),
            'DifficultyCompletingTasks': yn(request.POST['difficulty_tasks']),
            'PersonalityChanges': yn(request.POST['personality_changes']),
            'FamilyHistoryAlzheimers': yn(request.POST['family_history']),
            'Depression': yn(request.POST['depression']),
            'HeadInjury': yn(request.POST['head_injury']),
        }

        df = pd.DataFrame([input_data])
        model = trained_model['model']

        prediction = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1] * 100

        result = (
            f"Prediction Result: The patient has Alzheimer’s disease.\nProbability: {prob:.2f}%"
            if prediction == 1 else
            f"Prediction Result: The patient does not have Alzheimer’s disease.\nEstimated Risk: {prob:.2f}%"
        )

        return render(request, 'users/predict_form.html', {
            'prediction': result,
            'model_used': trained_model['name'],
            'risk_percent': round(prob, 2)
        })

    return render(request, 'users/predict_form.html')
