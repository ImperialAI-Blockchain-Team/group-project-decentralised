import web3 from './web3';
import { modelDatabase} from "../abi/abi";

const address = '0xEdFd5D8c18031C365Dae09e08da3E6C41344B327';

export default new web3.eth.Contract(modelDatabase, address);
