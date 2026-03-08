function show(id){

    document.querySelectorAll(".card")
    .forEach(c=>c.style.display="none")
    
    document.getElementById(id).style.display="block"
    
    }
    
    function signup(){
    
    let data=new FormData()
    
    data.append("username",sname.value)
    data.append("email",semail.value)
    data.append("password",spass.value)
    
    fetch("http://127.0.0.1:5000/signup",{
    method:"POST",
    body:data
    })
    .then(r=>r.json())
    .then(d=>alert(d.msg))
    
    }
    
    function login(){
    
    let data=new FormData()
    
    data.append("email",lemail.value)
    data.append("password",lpass.value)
    
    fetch("http://127.0.0.1:5000/login",{
    method:"POST",
    body:data
    })
    .then(r=>r.json())
    .then(d=>alert(d.msg))
    
    }
    
    function register(){
    
    let data=new FormData()
    
    data.append("name",name.value)
    data.append("location",location.value)
    data.append("email",email.value)
    data.append("image",photo.files[0])
    
    fetch("http://127.0.0.1:5000/register_missing",{
    method:"POST",
    body:data
    })
    .then(r=>r.json())
    .then(d=>alert(d.msg))
    
    }
    
    function find(){
    
    let data=new FormData()
    
    data.append("image",foundphoto.files[0])
    
    fetch("http://127.0.0.1:5000/find",{
    method:"POST",
    body:data
    })
    .then(r=>r.json())
    .then(d=>alert(d.msg))
    
    }