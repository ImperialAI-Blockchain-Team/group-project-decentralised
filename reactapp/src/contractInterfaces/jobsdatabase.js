import web3 from './web3';
import { jobs} from "../abi/abi";

const address = '0xB079b8793c9c4e2b6a5d71b859E7Fbb16C08629A';

export default new web3.eth.Contract(jobs, address);