//! 🖥️ **SELEÇÃO CENTRALIZADA DE DISPOSITIVO DE COMPUTAÇÃO**
//!
//! Ponto único de decisão sobre onde os tensores são computados: Metal GPU
//! (Apple Silicon) com fallback para CPU. Usado por todos os comandos da CLI
//! e pelo servidor web, para que a seleção de GPU seja consistente em todo
//! o projeto em vez de duplicada em cada comando.

use candle_core::Device;

/// Resolve o dispositivo de computação: tenta Metal GPU, cai para CPU se
/// indisponível.
///
/// `Device::new_metal` pode entrar em pânico em vez de retornar `Err` quando
/// o processo não enxerga nenhuma GPU via `MTLCopyAllDevices()` — um bug
/// conhecido do candle-core (ver https://github.com/huggingface/candle/issues/3566),
/// comum em execuções não-interativas/automatizadas sem sessão de janela.
/// Por isso o pânico é capturado aqui e tratado como "Metal indisponível"
/// em vez de derrubar o processo inteiro.
pub fn resolve_device() -> Device {
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(|| Device::new_metal(0));
    std::panic::set_hook(previous_hook);

    match result {
        Ok(Ok(device)) => {
            println!("🚀 Usando dispositivo: Metal GPU");
            device
        }
        Ok(Err(e)) => {
            println!("⚠️  Metal GPU não disponível ({}), usando CPU", e);
            Device::Cpu
        }
        Err(_) => {
            println!("⚠️  Metal GPU não pôde ser inicializada nesta sessão (sem acesso ao driver Metal), usando CPU");
            Device::Cpu
        }
    }
}
