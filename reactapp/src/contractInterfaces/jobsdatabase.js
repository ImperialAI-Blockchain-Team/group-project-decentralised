import web3 from './web3';
import { jobs} from "../abi/abi";

const address = '0x65a6DCe3ce74b409Adb1B31CC53Cd6c141A8c681';

export default new web3.eth.Contract(jobs, address);