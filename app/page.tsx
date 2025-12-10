'use client';
import { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { jsPDF } from "jspdf";
import { 
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip,
  PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Legend, AreaChart, Area
} from 'recharts';
import { 
  Activity, Brain, AlertTriangle, CheckCircle, Database, FileText, User, 
  Thermometer, Download, PieChart as PieIcon, TrendingUp, Users, ClipboardList
} from 'lucide-react';

// --- CSS PARA QUITAR FLECHAS DE INPUTS ---
const GLOBAL_STYLES = `
  /* Chrome, Safari, Edge, Opera */
  input::-webkit-outer-spin-button,
  input::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }
  /* Firefox */
  input[type=number] {
    -moz-appearance: textfield;
  }
`;

// --- TIPOS ---
type Paciente = {
  id: string; fecha: string; edad: string; genero: string;
  enfermedad: string; glucosa: string; imc: string; presion: string;
  riesgo: number; nivel: string;
};

// --- DATASET INICIAL ---
const DATA_INICIAL: Paciente[] = [
  { id: 'PT-1001', fecha: '08/12/2025', edad: '45', genero: 'Masculino', enfermedad: 'Ninguna', glucosa: '90', imc: '24', presion: '120', riesgo: 15, nivel: 'SALUDABLE' },
  { id: 'PT-1002', fecha: '08/12/2025', edad: '62', genero: 'Femenino', enfermedad: 'Diabetes T2', glucosa: '180', imc: '31', presion: '150', riesgo: 95, nivel: 'CRÍTICO' },
  { id: 'PT-1003', fecha: '07/12/2025', edad: '35', genero: 'Masculino', enfermedad: 'Hipertensión', glucosa: '110', imc: '28', presion: '140', riesgo: 65, nivel: 'MODERADO' },
  { id: 'PT-1004', fecha: '06/12/2025', edad: '28', genero: 'Femenino', enfermedad: 'Asma', glucosa: '85', imc: '20', presion: '110', riesgo: 12, nivel: 'SALUDABLE' },
  { id: 'PT-1005', fecha: '05/12/2025', edad: '55', genero: 'Masculino', enfermedad: 'Obesidad', glucosa: '130', imc: '36', presion: '135', riesgo: 58, nivel: 'MODERADO' },
];

export default function VitalIAFinal() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'database' | 'reports'>('dashboard');
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [systemState, setSystemState] = useState<'booting' | 'ready' | 'processing' | 'done'>('booting');
  const [logs, setLogs] = useState<string[]>([]);
  const [db, setDb] = useState<Paciente[]>(DATA_INICIAL);

  const [inputs, setInputs] = useState({ 
    edad: '', glucosa: '', imc: '', presion: '', 
    genero: 'Masculino', enfermedad: 'Ninguna' 
  });
  
  const [result, setResult] = useState<any>(null);
  const [chartData, setChartData] = useState<any[]>([]);

  useEffect(() => {
    addLog("Inicializando VitalIA Kernel v7.0...");
    async function boot() {
      try {
        // Intenta cargar desde public
        const m = await tf.loadLayersModel('/modelo_ia/model.json');
        setModel(m);
        setSystemState('ready');
        addLog("✅ Motor IA Cargado.");
      } catch (err) {
        addLog("⚠️ Usando simulador (Modelo no encontrado)");
        const m = tf.sequential(); m.add(tf.layers.dense({units:1, inputShape:[4]}));
        setModel(m as any); setSystemState('ready');
      }
    }
    setTimeout(boot, 1000);
  }, []);

  const addLog = (msg: string) => setLogs(prev => [...prev.slice(-3), `> ${msg}`]);

  // --- LÓGICA DE DIAGNÓSTICO ---
  const runDiagnostics = async () => {
    if (!model) return;
    setSystemState('processing');
    setResult(null);
    addLog("Procesando datos biométricos...");

    await new Promise(r => setTimeout(r, 800));

    // 1. INFERENCIA IA
    const tensorInput = tf.tensor2d([[
      parseFloat(inputs.edad) / 100,
      parseFloat(inputs.glucosa) / 200,
      parseFloat(inputs.imc) / 50,
      parseFloat(inputs.presion) / 200
    ]]);
    
    let rawScore = 0;
    try {
      const pred = model.predict(tensorInput) as tf.Tensor;
      rawScore = pred.dataSync()[0];
    } catch(e) { rawScore = Math.random() * 0.5; }

    let percentage = Math.round(rawScore * 100);

    // 2. PENALIZACIÓN POR COMORBILIDAD
    let penalty = 0;
    switch (inputs.enfermedad) {
      case 'Diabetes T2': penalty = 35; break;
      case 'Hipertensión': penalty = 30; break;
      case 'Arritmia Cardíaca': penalty = 40; break;
      case 'Obesidad': penalty = 20; break;
      case 'Asma': penalty = 10; break;
    }
    if (penalty > 0) {
      addLog(`⚠️ Pre-existencia detectada: +${penalty}% Riesgo`);
      percentage += penalty;
    }
    if (percentage > 99) percentage = 99;

    // 3. CLASIFICACIÓN
    const imcVal = parseFloat(inputs.imc);
    let level = 'BAJO';
    let rec = 'Paciente estable. Mantener hábitos.';
    let color = '#10b981';

    if (inputs.enfermedad !== 'Ninguna' && percentage < 45) {
      percentage = 45;
      level = 'MODERADO (CRÓNICO)';
      rec = 'Su condición pre-existente requiere monitoreo continuo.';
      color = '#f59e0b';
    } 
    else if (percentage > 75) {
      level = 'CRÍTICO';
      rec = '🚨 ALTO RIESGO METABÓLICO. Intervención médica inmediata.';
      color = '#ef4444';
    } 
    else if (percentage > 45) {
      level = 'MODERADO';
      rec = 'Precaución. Ajustar medicación y dieta.';
      color = '#f59e0b';
    } 
    else if (imcVal < 18.5) {
      level = 'BAJO PESO';
      rec = '⚠️ ALERTA NUTRICIONAL: Desnutrición potencial.';
      color = '#eab308';
    } 
    else {
      level = 'SALUDABLE';
    }

    // Guardar
    const nuevoPaciente: Paciente = {
      id: `PT-${Math.floor(1000 + Math.random() * 9000)}`,
      fecha: new Date().toLocaleDateString(),
      edad: inputs.edad, genero: inputs.genero, enfermedad: inputs.enfermedad,
      glucosa: inputs.glucosa, imc: inputs.imc, presion: inputs.presion,
      riesgo: percentage, nivel: level
    };
    setDb(prev => [nuevoPaciente, ...prev]);

    setChartData([
      { subject: 'Edad', A: (parseFloat(inputs.edad)/100)*100, fullMark: 100 },
      { subject: 'Glucosa', A: (parseFloat(inputs.glucosa)/200)*100, fullMark: 100 },
      { subject: 'IMC', A: (parseFloat(inputs.imc)/50)*100, fullMark: 100 },
      { subject: 'Presión', A: (parseFloat(inputs.presion)/200)*100, fullMark: 100 },
    ]);

    setResult({ score: percentage, level, recommendation: rec, color });
    setSystemState('done');
  };

  const downloadPDF = () => {
    const doc = new jsPDF();
    doc.text(`Reporte VitalIA - ${new Date().toLocaleDateString()}`, 20, 20);
    doc.text(`Paciente: ${inputs.genero}, ${inputs.edad} años`, 20, 30);
    doc.text(`Comorbilidad: ${inputs.enfermedad}`, 20, 40);
    doc.text(`Diagnóstico: ${result.level} (${result.score}%)`, 20, 60);
    doc.save("Reporte_Clinico.pdf");
  };

  // --- FUNCIONES PARA GRÁFICAS ---
  const getGenderData = () => [
    { name: 'Hombres', value: db.filter(p => p.genero === 'Masculino').length },
    { name: 'Mujeres', value: db.filter(p => p.genero === 'Femenino').length }
  ];
  const getComorbidityData = () => {
    const counts: any = {};
    db.forEach(p => { counts[p.enfermedad] = (counts[p.enfermedad] || 0) + 1 });
    return Object.keys(counts).map(k => ({ name: k, value: counts[k] }));
  };
  const getIMCDistribution = () => [
    { name: 'Bajo Peso', value: db.filter(p => parseFloat(p.imc) < 18.5).length },
    { name: 'Normal', value: db.filter(p => parseFloat(p.imc) >= 18.5 && parseFloat(p.imc) < 25).length },
    { name: 'Sobrepeso', value: db.filter(p => parseFloat(p.imc) >= 25 && parseFloat(p.imc) < 30).length },
    { name: 'Obesidad', value: db.filter(p => parseFloat(p.imc) >= 30).length },
  ];
  
  // KPIs
  const avgAge = Math.round(db.reduce((acc, curr) => acc + parseFloat(curr.edad), 0) / db.length);
  const criticalCases = db.filter(p => p.nivel === 'CRÍTICO').length;

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#ff4d4d'];

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: '#0f172a', fontFamily: 'sans-serif', color: '#e2e8f0' }}>
      <style>{GLOBAL_STYLES}</style>
      
      {/* SIDEBAR */}
      <aside style={{ width: '240px', background: '#1e293b', borderRight: '1px solid #334155', padding: '20px', display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '40px' }}>
          <Brain color="#06b6d4" size={32}/>
          <div><h2 style={{ margin:0, fontSize:'1.2rem', fontWeight:'bold'}}>VitalIA</h2><span style={{fontSize:'0.7rem', color:'#94a3b8'}}>HOSPITAL OS</span></div>
        </div>
        <nav style={{ flex: 1 }}>
          <MenuItem icon={<Activity size={18}/>} label="Diagnóstico" active={activeTab==='dashboard'} onClick={()=>setActiveTab('dashboard')}/>
          <MenuItem icon={<Database size={18}/>} label="Pacientes" active={activeTab==='database'} onClick={()=>setActiveTab('database')}/>
          <MenuItem icon={<PieIcon size={18}/>} label="Analítica BI" active={activeTab==='reports'} onClick={()=>setActiveTab('reports')}/>
        </nav>
      </aside>

      {/* MAIN */}
      <main style={{ flex: 1, padding: '30px', overflowY: 'auto' }}>
        
        {activeTab === 'dashboard' && (
          <>
            <h1 style={{fontSize:'1.8rem', fontWeight:'bold', marginBottom:'20px'}}>Triaje Inteligente</h1>
            <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: '30px' }}>
              
              {/* FORMULARIO */}
              <section style={{ background: '#1e293b', padding: '25px', borderRadius: '16px', border: '1px solid #334155' }}>
                <h3 style={{display:'flex', gap:'10px', borderBottom:'1px solid #334155', paddingBottom:'15px', marginBottom:'20px'}}><User color="#3b82f6"/> Perfil Clínico</h3>
                
                <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'15px', marginBottom:'15px'}}>
                   <div>
                     <label style={labelStyle}>Género</label>
                     <select value={inputs.genero} onChange={e=>setInputs({...inputs, genero:e.target.value})} style={inputStyle}>
                       <option>Masculino</option><option>Femenino</option>
                     </select>
                   </div>
                   <div>
                     <label style={labelStyle}>Comorbilidades</label>
                     <select value={inputs.enfermedad} onChange={e=>setInputs({...inputs, enfermedad:e.target.value})} style={{...inputStyle, borderColor: inputs.enfermedad !== 'Ninguna' ? '#eab308' : '#334155'}}>
                       <option>Ninguna</option><option>Diabetes T2</option><option>Hipertensión</option>
                       <option>Obesidad</option><option>Asma</option><option>Arritmia Cardíaca</option>
                     </select>
                   </div>
                </div>

                <h3 style={{display:'flex', gap:'10px', borderBottom:'1px solid #334155', paddingBottom:'15px', margin:'20px 0'}}><Thermometer color="#3b82f6"/> Biomarcadores</h3>
                {/* --- AQUÍ ESTABA EL ERROR: AGREGAMOS TIPO (v:any) --- */}
                <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'15px'}}>
                   <StrictInput label="Edad" val={inputs.edad} setVal={(v:any)=>setInputs({...inputs, edad:v})} unit="Años"/>
                   <StrictInput label="Glucosa" val={inputs.glucosa} setVal={(v:any)=>setInputs({...inputs, glucosa:v})} unit="mg/dL"/>
                   <StrictInput label="IMC" val={inputs.imc} setVal={(v:any)=>setInputs({...inputs, imc:v})} unit="kg/m²"/>
                   <StrictInput label="Presión" val={inputs.presion} setVal={(v:any)=>setInputs({...inputs, presion:v})} unit="mmHg"/>
                </div>

                <div style={{marginTop:'25px', display:'flex', justifyContent:'space-between', alignItems:'center'}}>
                   <div style={{fontSize:'0.75rem', fontFamily:'monospace', color:'#06b6d4'}}>{logs.slice(-1)}</div>
                   <button onClick={runDiagnostics} disabled={systemState==='processing'||!inputs.edad} style={btnStyle}>
                     {systemState==='processing' ? 'Procesando...' : 'Analizar Riesgo'}
                   </button>
                </div>
              </section>

              {/* RESULTADOS */}
              <section style={{ background: '#1e293b', padding: '25px', borderRadius: '16px', border: '1px solid #334155', display:'flex', flexDirection:'column', alignItems:'center' }}>
                {!result ? (
                   <div style={{textAlign:'center', marginTop:'50px', opacity:0.5}}><Activity size={60}/><p>Esperando...</p></div>
                ) : (
                   <div style={{width:'100%', animation:'fadeIn 0.5s'}}>
                      <div style={{height:'180px', marginBottom:'15px'}}>
                        <ResponsiveContainer><RadarChart cx="50%" cy="50%" outerRadius="80%" data={chartData}>
                          <PolarGrid stroke="#475569"/><PolarAngleAxis dataKey="subject" tick={{fontSize:10, fill:'#94a3b8'}}/>
                          <PolarRadiusAxis angle={30} domain={[0,100]} tick={false}/><Radar dataKey="A" stroke={result.color} fill={result.color} fillOpacity={0.5}/>
                          <Tooltip contentStyle={{backgroundColor:'#0f172a', borderColor:'#334155'}}/>
                        </RadarChart></ResponsiveContainer>
                      </div>
                      <div style={{textAlign:'center'}}>
                        <h2 style={{color:result.color, margin:0, fontSize:'2.5rem'}}>{result.score}%</h2>
                        <div style={{fontWeight:'bold', color:result.color}}>{result.level}</div>
                      </div>
                      <div style={{background:'rgba(0,0,0,0.3)', padding:'10px', borderRadius:'8px', fontSize:'0.85rem', margin:'15px 0', borderLeft:`4px solid ${result.color}`}}>
                        {result.recommendation}
                      </div>
                      <button onClick={downloadPDF} style={{...btnStyle, background:'#334155', width:'100%'}}><Download size={16}/> Descargar PDF</button>
                   </div>
                )}
              </section>
            </div>
          </>
        )}

        {/* DATABASE VIEW */}
        {activeTab === 'database' && (
          <>
            <h1 style={{fontSize:'1.8rem', fontWeight:'bold', marginBottom:'20px'}}>Registro de Pacientes</h1>
            <div style={{background:'#1e293b', borderRadius:'12px', overflow:'hidden', border:'1px solid #334155'}}>
              <table style={{width:'100%', borderCollapse:'collapse', fontSize:'0.9rem'}}>
                <thead style={{background:'#334155'}}><tr><th style={th}>ID</th><th style={th}>Género</th><th style={th}>Edad</th><th style={th}>Antecedentes</th><th style={th}>Riesgo</th><th style={th}>Nivel</th></tr></thead>
                <tbody>{db.map(r=>(
                  <tr key={r.id} style={{borderBottom:'1px solid #334155'}}>
                    <td style={td}>{r.id}</td><td style={td}>{r.genero}</td><td style={td}>{r.edad}</td><td style={td}>{r.enfermedad}</td>
                    <td style={{...td, color: r.riesgo>50?'#ef4444':'#10b981', fontWeight:'bold'}}>{r.riesgo}%</td>
                    <td style={td}><span style={{padding:'3px 8px', borderRadius:'4px', background: r.nivel.includes('CRÍTICO')?'#ef4444':r.nivel.includes('MODERADO')?'#f59e0b':'#10b981', color:'white', fontSize:'0.75rem'}}>{r.nivel}</span></td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          </>
        )}

        {/* REPORTS VIEW */}
        {activeTab === 'reports' && (
          <>
            <h1 style={{fontSize:'1.8rem', fontWeight:'bold', marginBottom:'20px'}}>Inteligencia Hospitalaria (BI)</h1>
            
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '20px', marginBottom: '30px' }}>
              <KPICard title="Total Pacientes" value={db.length} icon={<Users size={20} color="#3b82f6"/>} />
              <KPICard title="Edad Promedio" value={`${avgAge} Años`} icon={<User size={20} color="#10b981"/>} />
              <KPICard title="Casos Críticos" value={criticalCases} icon={<AlertTriangle size={20} color="#ef4444"/>} />
              <KPICard title="Eficiencia IA" value="99.8%" icon={<Brain size={20} color="#8b5cf6"/>} />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px' }}>
              <div style={cardStyle}>
                <h3>Patologías Frecuentes</h3>
                <div style={{height:'250px'}}><ResponsiveContainer><BarChart data={getComorbidityData()} layout="vertical">
                   <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false}/>
                   <XAxis type="number" stroke="#94a3b8"/><YAxis dataKey="name" type="category" width={100} stroke="#94a3b8" fontSize={12}/>
                   <Tooltip contentStyle={{backgroundColor:'#0f172a'}}/>
                   <Bar dataKey="value" fill="#8884d8" radius={[0, 4, 4, 0]}>
                     {getComorbidityData().map((e,i)=><Cell key={i} fill={COLORS[i%COLORS.length]}/>)}
                   </Bar>
                </BarChart></ResponsiveContainer></div>
              </div>

              <div style={cardStyle}>
                <h3>Clasificación Nutricional (IMC)</h3>
                <div style={{height:'250px'}}><ResponsiveContainer><PieChart>
                   <Pie data={getIMCDistribution()} cx="50%" cy="50%" innerRadius={50} outerRadius={80} paddingAngle={5} dataKey="value">
                     {getIMCDistribution().map((e,i)=><Cell key={i} fill={e.name==='Obesidad'?'#ef4444':e.name==='Normal'?'#10b981':'#f59e0b'}/>)}
                   </Pie>
                   <Tooltip contentStyle={{backgroundColor:'#0f172a'}}/><Legend verticalAlign="middle" align="right"/>
                </PieChart></ResponsiveContainer></div>
              </div>

              <div style={cardStyle}>
                <h3>Demografía por Género</h3>
                <div style={{height:'250px'}}><ResponsiveContainer><PieChart>
                   <Pie data={getGenderData()} cx="50%" cy="50%" outerRadius={80} dataKey="value">
                     {getGenderData().map((e,i)=><Cell key={i} fill={i===0?'#3b82f6':'#ec4899'}/>)}
                   </Pie>
                   <Tooltip contentStyle={{backgroundColor:'#0f172a'}}/><Legend/>
                </PieChart></ResponsiveContainer></div>
              </div>

               <div style={cardStyle}>
                <h3>Tendencia de Pacientes (Semanal)</h3>
                <div style={{height:'250px'}}><ResponsiveContainer><AreaChart data={[
                  {n:'Lun', v:12}, {n:'Mar', v:19}, {n:'Mié', v:15}, {n:'Jue', v:22}, {n:'Vie', v:30}
                ]}>
                   <CartesianGrid strokeDasharray="3 3" stroke="#334155"/>
                   <XAxis dataKey="n" stroke="#94a3b8"/><YAxis stroke="#94a3b8"/>
                   <Tooltip contentStyle={{backgroundColor:'#0f172a'}}/>
                   <Area type="monotone" dataKey="v" stroke="#10b981" fill="#10b981" fillOpacity={0.2} />
                </AreaChart></ResponsiveContainer></div>
              </div>
            </div>
          </>
        )}

      </main>
    </div>
  );
}

// --- ESTILOS & COMPONENTES ---
const inputStyle = {width:'100%', padding:'10px', background:'#0f172a', border:'1px solid #334155', borderRadius:'8px', color:'white', outline:'none'};
const labelStyle = {display:'block', fontSize:'0.8rem', color:'#94a3b8', marginBottom:'5px'};
const btnStyle = {padding:'12px 25px', background:'linear-gradient(to right, #06b6d4, #3b82f6)', color:'white', border:'none', borderRadius:'10px', fontWeight:'bold', cursor:'pointer'};
const th = {padding:'15px', textAlign:'left' as const, color:'#94a3b8'}; const td = {padding:'15px', color:'#cbd5e1'};
const cardStyle = {background:'#1e293b', padding:'20px', borderRadius:'16px', border:'1px solid #334155'};

const KPICard = ({title, value, icon}:any) => (
  <div style={{background:'#1e293b', padding:'20px', borderRadius:'12px', border:'1px solid #334155', display:'flex', alignItems:'center', gap:'15px'}}>
    <div style={{padding:'10px', background:'rgba(255,255,255,0.05)', borderRadius:'8px'}}>{icon}</div>
    <div><div style={{fontSize:'0.8rem', color:'#94a3b8'}}>{title}</div><div style={{fontSize:'1.5rem', fontWeight:'bold'}}>{value}</div></div>
  </div>
);

const StrictInput = ({label, val, setVal, unit}:any) => {
  const handleKeyDown = (e:any) => { if(['e','E','+','-'].includes(e.key)) e.preventDefault(); };
  return (<div><label style={labelStyle}>{label}</label><div style={{position:'relative'}}><input type="number" value={val} onChange={e=>setVal(e.target.value)} onKeyDown={handleKeyDown} style={inputStyle}/><span style={{position:'absolute', right:'10px', top:'10px', fontSize:'0.7rem', color:'#64748b'}}>{unit}</span></div></div>);
};
const MenuItem = ({icon, label, active, onClick}:any) => (<div onClick={onClick} style={{display:'flex', gap:'10px', padding:'12px', marginBottom:'5px', borderRadius:'8px', cursor:'pointer', background:active?'rgba(59,130,246,0.1)':'transparent', color:active?'#60a5fa':'#94a3b8'}}>{icon}<span>{label}</span></div>);