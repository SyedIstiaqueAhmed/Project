function handleButton(){
    hideElementById('home-screen');
    showElementById('home-screen2')
}
function imageClassification(){
    hideElementById('home-screen2');
    showElementById('image')
}
function wavClassification(){
    hideElementById('home-screen2');
    showElementById('wav')
}
function audioClassification(){
    hideElementById('home-screen2');
    showElementById('audio')
}
function imageClassification2(){
    hideElementById('image');
    showElementById('home-screen2')
}
function wavClassification2(){
    hideElementById('wav');
    showElementById('home-screen2')
}
function audioClassification2(){
    hideElementById('audio');
    showElementById('home-screen2')
}
function hideElementById(elementId){
    const element = document.getElementById(elementId); 
    element.classList.add('hidden');

}
function showElementById(elementId){
    const element = document.getElementById(elementId); 
    element.classList.remove('hidden');

}