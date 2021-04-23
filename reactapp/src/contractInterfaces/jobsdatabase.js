import web3 from './web3';
import { jobs} from "../abi/abi";

const address = '0xD1a210292F6D37098114AFF851D747Ba6ccBAB9B';

export default new web3.eth.Contract(jobs, address);
