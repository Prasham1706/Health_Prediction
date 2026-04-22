// Healthcare Risk Prediction - Frontend JavaScript
// Connects to ML backend API for real predictions

const API_URL = window.location.origin;

// Smooth scroll for navigation
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});

// Navbar scroll effect
window.addEventListener('scroll', () => {
  const navbar = document.getElementById('navbar');
  if (window.scrollY > 50) {
    navbar.style.background = 'rgba(10, 10, 30, 0.95)';
    navbar.style.backdropFilter = 'blur(10px)';
  } else {
    navbar.style.background = 'rgba(10, 10, 30, 0.8)';
  }
});

// Form submission handler
const form = document.getElementById('assessmentForm');
const resultsSection = document.getElementById('results');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  
  // Show loading state
  const submitButton = form.querySelector('button[type="submit"]');
  const originalText = submitButton.textContent;
  submitButton.textContent = 'Analyzing...';
  submitButton.disabled = true;
  
  try {
    // Collect form data using a more robust method
    const data = {};
    const formData = new FormData(form);
    
    // Convert FormData to a plain object
    for (const [key, value] of formData.entries()) {
      data[key] = value;
    }
    
    // Explicitly handle checkboxes (which are often 'on' or missing)
    const checkboxes = form.querySelectorAll('input[type="checkbox"]');
    checkboxes.forEach(cb => {
      data[cb.name || cb.id] = cb.checked;
    });
    
    console.log('Processed Form Data:', data);
    
    // Calculate BMI
    const height = parseFloat(data.height) / 100; // convert cm to m
    const weight = parseFloat(data.weight);
    const bmi = weight / (height * height);
    
    // Map smoking status to numeric value
    const smokingMap = {
      'never': 0,
      'former': 3,
      'current': 2
    };
    
    // Prepare diabetes prediction data
    const diabetesData = {
      gender: data.gender === 'male' ? 1 : 0,
      age: parseInt(data.age),
      hypertension: parseInt(data.systolic) > 140 || parseInt(data.diastolic) > 90 ? 1 : 0,
      heart_disease: 0, // We'll update this based on family history
      smoking_history: smokingMap[data.smoking] || 0,
      bmi: bmi,
      HbA1c_level: parseFloat(data.hba1c || 5.7),
      blood_glucose_level: parseInt(data.glucose)
    };
    
    // Prepare heart disease prediction data
    const heartData = {
      Chest_Pain: data.chestPain ? 1 : 0,
      Shortness_of_Breath: data.breathShortness ? 1 : 0,
      Fatigue: data.fatigue ? 1 : 0,
      Palpitations: data.palpitations ? 1 : 0,
      Dizziness: data.dizziness ? 1 : 0,
      Swelling: data.swelling ? 1 : 0,
      Pain_Arms_Jaw_Back: data.painRadiation ? 1 : 0,
      Cold_Sweats_Nausea: data.nauseaSweats ? 1 : 0,
      High_BP: parseInt(data.systolic) > 140 || parseInt(data.diastolic) > 90 ? 1 : 0,
      High_Cholesterol: parseInt(data.cholesterol) > 240 ? 1 : 0,
      Diabetes: parseFloat(data.hba1c) > 6.5 || parseInt(data.glucose) > 126 ? 1 : 0,
      Smoking: data.smoking === 'current' ? 1 : 0,
      Obesity: bmi > 30 ? 1 : 0,
      Sedentary_Lifestyle: parseFloat(data.exercise) < 2 ? 1 : 0,
      Family_History: data.familyHeartDisease ? 1 : 0,
      Chronic_Stress: data.chronicStress ? 1 : 0,
      Gender: data.gender === 'male' ? 1 : 0,
      Age: parseInt(data.age)
    };
    
    // Update heart disease based on family history
    if (data.familyHeartDisease) {
      diabetesData.heart_disease = 1;
    }
    
    // Make API calls
    const [diabetesResponse, heartResponse] = await Promise.all([
      fetch(`${API_URL}/predict/diabetes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(diabetesData)
      }),
      fetch(`${API_URL}/predict/heart-disease`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(heartData)
      })
    ]);
    
    if (!diabetesResponse.ok || !heartResponse.ok) {
      throw new Error('API request failed');
    }
    
    const diabetesResult = await diabetesResponse.json();
    const heartResult = await heartResponse.json();
    
    // Calculate overall risk
    const overallRisk = (diabetesResult.risk_probability + heartResult.risk_probability) / 2;
    
    // Display results
    displayResults({
      overall: overallRisk * 100,
      diabetes: diabetesResult.risk_percentage,
      heart: heartResult.risk_percentage,
      cardiovascular: heartResult.risk_percentage,
      metabolic: diabetesResult.risk_percentage,
      lifestyle: ((data.smoking === 'current' ? 30 : 0) + (parseFloat(data.exercise) < 2 ? 20 : 0)) / 2,
      genetic: (data.familyHeartDisease ? 25 : 0) + (data.familyDiabetes ? 25 : 0)
    });
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    
  } catch (error) {
    console.error('Error:', error);
    alert('Error connecting to the prediction service. Please make sure the backend server is running (python app.py)');
  } finally {
    submitButton.textContent = originalText;
    submitButton.disabled = false;
  }
});

function displayResults(risks) {
  // Show results section
  resultsSection.style.display = 'block';
  
  // Overall risk score
  const riskPercentage = document.getElementById('riskPercentage');
  const riskCategory = document.getElementById('riskCategory');
  
  animateValue(riskPercentage, 0, Math.round(risks.overall), 1500, '%');
  
  // Determine risk category
  let category, color;
  if (risks.overall < 30) {
    category = 'Low Risk';
    color = '#10b981';
  } else if (risks.overall < 60) {
    category = 'Moderate Risk';
    color = '#f59e0b';
  } else {
    category = 'High Risk';
    color = '#ef4444';
  }
  
  riskCategory.textContent = category;
  riskCategory.style.color = color;
  
  // Update risk factors
  updateRiskFactor('cardio', risks.cardiovascular);
  updateRiskFactor('metabolic', risks.metabolic);
  updateRiskFactor('lifestyle', risks.lifestyle);
  updateRiskFactor('genetic', risks.genetic);
  
  // Generate recommendations
  generateRecommendations(risks);
}

function updateRiskFactor(id, percentage) {
  const valueElement = document.getElementById(`${id}Value`);
  const progressElement = document.getElementById(`${id}Progress`);
  
  animateValue(valueElement, 0, Math.round(percentage), 1000, '%');
  
  setTimeout(() => {
    progressElement.style.width = `${percentage}%`;
    
    // Color based on risk level
    if (percentage < 30) {
      progressElement.style.background = 'linear-gradient(90deg, #10b981, #34d399)';
    } else if (percentage < 60) {
      progressElement.style.background = 'linear-gradient(90deg, #f59e0b, #fbbf24)';
    } else {
      progressElement.style.background = 'linear-gradient(90deg, #ef4444, #f87171)';
    }
  }, 100);
}

function animateValue(element, start, end, duration, suffix = '') {
  const range = end - start;
  const increment = range / (duration / 16);
  let current = start;
  
  const timer = setInterval(() => {
    current += increment;
    if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
      current = end;
      clearInterval(timer);
    }
    element.textContent = Math.round(current) + suffix;
  }, 16);
}

function generateRecommendations(risks) {
  const recommendationsList = document.getElementById('recommendationsList');
  const recommendations = [];
  
  if (risks.cardiovascular > 50) {
    recommendations.push({
      icon: '❤️',
      title: 'Cardiovascular Health',
      text: 'Consider consulting a cardiologist for a comprehensive heart health assessment.'
    });
  }
  
  if (risks.metabolic > 50) {
    recommendations.push({
      icon: '🩺',
      title: 'Metabolic Health',
      text: 'Monitor blood sugar levels regularly and consider dietary modifications to reduce diabetes risk.'
    });
  }
  
  if (risks.lifestyle > 40) {
    recommendations.push({
      icon: '🏃',
      title: 'Lifestyle Changes',
      text: 'Increase physical activity to at least 150 minutes per week and consider smoking cessation programs.'
    });
  }
  
  if (risks.genetic > 30) {
    recommendations.push({
      icon: '🧬',
      title: 'Genetic Factors',
      text: 'Given your family history, schedule regular health screenings and maintain preventive care.'
    });
  }
  
  // Always add general recommendations
  recommendations.push({
    icon: '🥗',
    title: 'Nutrition',
    text: 'Adopt a balanced diet rich in fruits, vegetables, whole grains, and lean proteins.'
  });
  
  recommendations.push({
    icon: '😴',
    title: 'Sleep & Stress',
    text: 'Aim for 7-9 hours of quality sleep and practice stress management techniques.'
  });
  
  // Render recommendations
  recommendationsList.innerHTML = recommendations.map(rec => `
    <div class="recommendation-item">
      <div class="recommendation-icon">${rec.icon}</div>
      <div class="recommendation-content">
        <h4>${rec.title}</h4>
        <p>${rec.text}</p>
      </div>
    </div>
  `).join('');
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  resultsSection.style.display = 'none';
});
