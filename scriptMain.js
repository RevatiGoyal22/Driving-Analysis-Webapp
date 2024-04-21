//date section
const currentDate = new Date();
const formattedDate = `${currentDate.getDate()}-${currentDate.getMonth()+1}-${currentDate.getFullYear()}`;
console.log(formattedDate);
document.getElementById('date').innerText=formattedDate;

